import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn

import models.anynet

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--export', action='store_true')


args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls


def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=0, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=0, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    if args.export:
        model.eval()
        export(model.module)
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    if args.evaluate:
        test(TestImgLoader, model, log)
        return

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 ==0:
            test(TestImgLoader, model, log)

    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    time_costs = []

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        if batch_idx == 0:    
            export(model.module, imgL, imgR)
            return

        with torch.no_grad():
            start = time.perf_counter()
            outputs = model(imgL, imgR)
            end = time.perf_counter()

            time_costs.append(end - start)
            # for x in range(stages):
            #     output = torch.squeeze(outputs[x], 1)
            #     D1s[x].update(error_estimating(output, disp_L).item())

            import cv2
            import numpy as np

            def un_norm(image_tensor):
                npimg = image_tensor.detach().cpu().numpy()
                npimg = np.transpose(npimg, (1,2,0))*255
                npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
                return npimg
            
            disp = outputs[-1].detach().cpu().numpy().squeeze()
            disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            Q = [[ 1., 0., 0., -2.7204545021057129e+02], [0., 1., 0., -3.0039989852905273e+02], 
                    [0., 0., 0., 5.0782172034846172e+02], [0., 0., 4.6879901308862252e+01, 0. ]]
            Q = np.array(Q)
            points_3d = cv2.reprojectImageTo3D(disp, Q) * 1000
            depth_map = points_3d[:, :, -1]
            depth_map = np.clip(depth_map, 0, 20000)
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

            left_image = un_norm(imgL[0])
            left_image = cv2.normalize(left_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
            cv2.putText(left_image, "origin", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disp_vis, "dispraity", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(depth_map, "depth", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(f"{args.save_path}/{batch_idx}.png", np.hstack([left_image, disp_vis, depth_map]))

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)
    print(sum(time_costs) / len(time_costs))


def export(model, inputL=None, inputR=None):
    import onnx
    from torch.onnx import register_custom_op_symbolic
    import torch.onnx.symbolic_helper as sym_help

    def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
        # mode
        #   'bilinear'      : onnx::Constant[value={0}]
        #   'nearest'       : onnx::Constant[value={1}]
        #   'bicubic'       : onnx::Constant[value={2}]
        # padding_mode
        #   'zeros'         : onnx::Constant[value={0}]
        #   'border'        : onnx::Constant[value={1}]
        #   'reflection'    : onnx::Constant[value={2}]
        mode = sym_help._maybe_get_const(mode, "i")
        padding_mode = sym_help._maybe_get_const(padding_mode, "i")
        mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
        padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
        align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

        return g.op("com.microsoft::GridSample", input, grid,
                    mode_s=mode_str,
                    padding_mode_s=padding_mode_str,
                    align_corners_i=align_corners)
        
    register_custom_op_symbolic('::grid_sampler', grid_sampler, 1)

    device = torch.device('cpu')
    model = model.to(device)

    if inputL is None:
        dummyL = torch.randn((1, 3, 640, 480)).to(device)
    else:
        dummyL = inputL.to(device)
        if dummyL.shape[0] > 1:
            dummyL = dummyL[0:1]
    
    if inputR is None:
        dummyR = torch.randn((1, 3, 640, 480)).to(device)
    else:
        dummyR = inputR.to(device)
        if dummyR.shape[0] > 1:
            dummyR = dummyR[0:1]

    onnx_path = "anynet_480x640.onnx"
    torch.onnx.export(
        model,
        (dummyL, dummyR),
        onnx_path,
        opset_version=11,
        input_names=["imgL", "imgR"],
        output_names=["disp1", "disp2", "disp3"]
    )

    onnx.checker.check_model(onnx.load(onnx_path))
    print("ONNX model exported to: " + onnx_path)

    import onnxruntime as ort
    import numpy as np
    dummy_output = model(dummyL, dummyR)[-1]
    sess = ort.InferenceSession(onnx_path)
    onnx_output = sess.run(None, {"imgL": dummyL.detach().cpu().numpy(), "imgR": dummyR.detach().cpu().numpy()})[-1]
    print(np.allclose(dummy_output.detach().cpu().numpy(), onnx_output))
    print(np.mean(np.abs(dummy_output.detach().cpu().numpy() - onnx_output)))


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
