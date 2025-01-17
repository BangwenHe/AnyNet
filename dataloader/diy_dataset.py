import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, log, split_file):
    left_fold = 'imgL/'
    right_fold = 'imgR/'
    disp_L = 'disp'

    train = [x for x in os.listdir(os.path.join(filepath, 'training', left_fold)) if is_image_file(x)]
    left_train = sorted([os.path.join(filepath, 'training', left_fold, img) for img in train])
    right_train = sorted([os.path.join(filepath, 'training', right_fold, img) for img in train])
    left_train_disp = sorted([os.path.join(filepath, 'training', disp_L, img) for img in train])

    val = [x for x in os.listdir(os.path.join(filepath, 'validation', left_fold)) if is_image_file(x)]
    left_val = sorted([os.path.join(filepath, 'validation', left_fold, img) for img in val])
    right_val = sorted([os.path.join(filepath, 'validation', right_fold, img) for img in val])
    left_val_disp = sorted([os.path.join(filepath, 'validation', disp_L, img) for img in val])

    return left_train, right_train, left_train_disp, left_val, right_val, left_val_disp
