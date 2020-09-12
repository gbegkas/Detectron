import os
import cv2
import numpy as np


def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def resize(directory, output_dir):
    mkdir(os.path.join(output_dir, 'shapes'))
    mkdir(os.path.join(output_dir, 'annotations'))
    for root, _, files in os.walk(os.path.join(directory, 'shapes')):
        for file in files:
            ann_file = os.path.join(directory, 'annotations', file[:-4] + '_mass.png')
            im = cv2.imread(os.path.join(root, file))
            ann = cv2.imread(ann_file,0)
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            res_im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR
            )
            im_file = os.path.join(output_dir, 'shapes', file)
            print(im_file)
            cv2.imwrite(im_file, res_im)

            res_ann = cv2.resize(
                ann,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR
            )
            ann_out = os.path.join(output_dir, 'annotations', os.path.basename(ann_file))
            print(ann_out)
            cv2.imwrite(ann_out, res_ann)


input_dir = '/mnt/genesis/vegas/Databases/CBIS-DDSM'
output = '/mnt/genesis/vegas/Databases/CBIS-DDSM_resized'
output_train = os.path.join(output, 'Train')
output_validation = os.path.join(output, 'Validation')
output_test = os.path.join(output, 'Test')
target_size = 800
max_size = 1333

mkdir(output)
mkdir(output_train)
mkdir(output_test)
mkdir(output_validation)

resize(os.path.join(input_dir, 'Train'), output_train)
resize(os.path.join(input_dir, 'Validation'), output_validation)
resize(os.path.join(input_dir, 'Test'), output_test)
