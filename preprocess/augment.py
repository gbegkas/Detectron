import os
import cv2
import numpy as np
from random import uniform
from detectron.datasets.json_dataset import JsonDataset
from keras.preprocessing import image


try:
    import scipy
    from scipy import ndimage

except ImportError:
    scipy = None


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


# input_dir = '/mnt/genesis/vegas/Databases/CBIS-DDSM_resized/Train/Ground Truth'
# output = '/mnt/genesis/vegas/transformed'
# test = '/mnt/genesis/vegas'

input_dir = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM/Train/Ground Truth'
# output = '/mnt/genesis/vegas/transformed'
test = '/home/gru'


ds = JsonDataset('CBIS_DDSM_train')
roidb = ds.get_roidb(gt=True,
                     proposal_file=None,
                     crowd_filter_thresh=0.7
                     )

# if not os.path.isdir(output):
#     os.mkdir(output)

images = [os.path.join(root, file) for root, _, files in os.walk(os.path.join(input_dir, 'Ground Truth')) for file in files]
for i in range(len(roidb)):
    if '00465_LEFT_CC' not in roidb[i]['image']:
        continue
    # angle = uniform(-20, 20)
    angle = 17.87189165364299
    # shear = uniform(-0.2, 0.2)
    shear = 0.18923481623384603
    # zoom = uniform(0.8, 1.2)
    zoom = 1.2
    channel_shift = uniform(-0.1, 0.1)
    # print('angle = '+str(angle))
    # print('shear = ' + str(shear))
    # print('zoom = ' + str(zoom))
    # print('channel shift = ' + str(channel_shift))

    # print(roidb[i]['image'])
    if i%10:
        print(str(i)+'/'+str(len(roidb)))

    img = cv2.imread(os.path.join(input_dir, os.path.basename(roidb[i]['image'])))

    # Debug border
    img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 3)

    img = image.apply_channel_shift(img, channel_shift).astype('uint8')
    h, w, _ = img.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    R = np.vstack((R, [0, 0, 1]))

    cos = np.abs(R[0, 0])
    sin = np.abs(R[0, 1])

    nW = int(((h * sin) + (w * cos)))
    nH = int(((h * cos) + (w * sin)))

    R[0, 2] += (nW / 2) - w / 2 + int(-np.sin(np.deg2rad(shear)) * w)
    R[1, 2] += (nH / 2) - h / 2 + int(np.sin(np.deg2rad(shear)) * h)
    # R[0, 2] *= zoom
    # R[1, 2] *= zoom

    nW += int(-np.sin(np.deg2rad(shear)) * nW)
    nH += int(np.sin(np.deg2rad(shear)) * nH)

    # nW = int(nW * zoom)
    # nH = int(nH * zoom)

    im_size_min = np.min([nW, nH])
    im_size_max = np.max([nW, nH])
    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)

    R[0, 2] *= im_scale
    R[1, 2] *= im_scale

    nW = int(nW * im_scale)
    nH = int(nH * im_scale)

    Scale_x = np.asarray([[im_scale, 0, 0], [0, 1, 0], [0, 0, 1]])
    Scale_y = np.asarray([[1, 0, 0], [0, im_scale, 0], [0, 0, 1]])

    Scale = np.dot(Scale_x, Scale_y)
    # T = np.dot(R, Scale)

    # Sx = np.asarray([[1, shear, 0],
    #                  [0, 1, 0],
    #                  [0, 0, 1]])
    #
    # Sy = np.asarray([[1, 0, 0],
    #                  [shear, 1, 0],
    #                  [0, 0, 1]])
    #
    # S = np.dot(Sx, Sy)

    S = np.array([[1, -np.sin(np.deg2rad(shear)), 0],
                  [0, np.cos(np.deg2rad(shear)), 0],
                  [0, 0, 1]])
    T = np.dot(R, S)
    T = np.dot(T, Scale)
    # Z = np.asarray([[zoom, 0, 0],
    #                 [0, zoom, 0],
    #                 [0, 0, 1]])

    # T = np.dot(np.dot(T, S), Z)

    img_transformed = cv2.warpPerspective(img, T, (nW, nH), borderMode=cv2.BORDER_REPLICATE)


    # img_transformed = cv2.resize(
    #             img_transformed,
    #             None,
    #             None,
    #             fx=im_scale,
    #             fy=im_scale,
    #             interpolation=cv2.INTER_LINEAR
    #         )

    for box in roidb[i]['boxes']:
        bx1, by1, bx2, by2 = box
        bw = (bx2 - bx1)
        bh = (by2 - by1)
        bcx, bcy = int(bx1 + bw / 2), int(by1 + bh / 2)

        ncenter = np.dot(T, [[bcx], [bcy], [1.0]])
        nbx = int(ncenter[0] - bw * im_scale / 2)
        nby = int(ncenter[1] - bh * im_scale / 2)
        nbx2 = int(nbx + bw * im_scale)
        nby2 = int(nby + bh * im_scale)

        cv2.rectangle(img_transformed, (nbx, nby), (nbx2, nby2), (0, 0, 255), 1)

    cv2.imwrite(os.path.join(test, os.path.basename(roidb[i]['image'])), img_transformed)
