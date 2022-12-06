import argparse
from itertools import product
import os

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image

import models.anynet


def parse_args():
    parser = argparse.ArgumentParser(description='inference for AnyNet')

    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--datapath', default=None, help='datapath')
    parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                        help='the path of saving checkpoints and log')
    parser.add_argument('--debug', action='store_true')
    
    # model config
    parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
    parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
    parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
    parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
    parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
    parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
    parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')

    parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                        help='pretrained model path')
    parser.add_argument('--calib_filepath', type=str, default='calib_result/mate40pro.yml') 

    args = parser.parse_args()
    return args


def build_input_images(image_folder):
    images = [i for i in os.listdir(image_folder) if i.endswith(("jpg", "png"))]
    left_images = [os.path.join(image_folder, i) for i in images if i.__contains__("left")]
    right_images = [os.path.join(image_folder, i) for i in images if i.__contains__("right")]

    left_images.sort()
    right_images.sort()

    assert len(left_images) == len(right_images)
    return left_images, right_images


def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


def un_norm(image_tensor):
    npimg = image_tensor.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1,2,0))*255
    npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    return npimg


def main(args):
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    model = models.anynet.AnyNet(args)
    model = torch.nn.DataParallel(model)
    assert os.path.exists(args.pretrained)

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    model.to(device)
    
    min_depth = 0
    max_depth = 8
    Q = load_stereo_coefficients(args.calib_filepath)[-1]

    left_image_paths, right_image_paths = build_input_images(args.datapath)

    for _, (left_image_path, right_image_path) in enumerate(zip(left_image_paths, right_image_paths)):
        imgL = preprocess(left_image_path).unsqueeze(0).to(device)
        imgR = preprocess(right_image_path).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(imgL, imgR)

        disp = outputs[-1].detach().cpu().numpy().squeeze()

        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_vis = 255 - disp_vis
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

        points_3d = cv2.reprojectImageTo3D(disp, Q)
        depth_map = points_3d[:, :, -1]
        depth_map = np.clip(depth_map, min_depth, max_depth)
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_map = 255 - depth_map
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        left_image = un_norm(imgL[0])
        left_image = cv2.normalize(left_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)

        if args.debug:
            cv2.imwrite(os.path.join(args.save_path, os.path.basename(left_image_path)), 
                np.hstack([left_image, disp_vis, depth_map]))
        else:
            cv2.imwrite(os.path.join(args.save_path, os.path.basename(left_image_path)), depth_map)
            # np.save(os.path.join(args.save_path, os.path.basename(left_image_path).replace(".png", ".npy")), points_3d[:, :, -1])


if __name__ == "__main__":
    args = parse_args()
    main(args)
