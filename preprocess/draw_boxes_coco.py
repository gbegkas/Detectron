import os
import cv2
import pycocotools as coco
from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# dataDir = '/mnt/genesis/vegas/Databases/CBIS-DDSM/Validation'
# annFile = '/mnt/genesis/vegas/Databases/CBIS-DDSM/Validation/instances_shape_val2019.json'

dataDir = '/mnt/Cargo_2/Diploma_Thesis/Databases/INbreast/Test'
annFile = '/mnt/Cargo_2/Diploma_Thesis/Databases/INbreast/Test/instances_shape_test2019.json'

if not os.path.isdir(os.path.join(dataDir, 'Ground Truth')):
    os.mkdir(os.path.join(dataDir, 'Ground Truth'))

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n', ' '.join(nms))
print('\n')
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n', ' '.join(nms))

catIds = coco.getCatIds(catNms=['mass'])
imgIds = coco.getImgIds(catIds=catIds)
c = 0
for imgId in imgIds:
    c += 1
    img = coco.loadImgs(imgId)[0]
    I = cv2.imread(os.path.join(dataDir, 'shapes', img['file_name']), 0)
    I = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)

    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)

    for ann in anns:
        x1 = int(ann['bbox'][0])
        y1 = int(ann['bbox'][1])
        x2 = x1 + int(ann['bbox'][2])
        y2 = y1 + int(ann['bbox'][3])

        cv2.rectangle(I, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # mask = coco.annToRLE(ann)
        # mask = coco.annToMask(ann)
        # plt.axis('off')
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # contours, h = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(I, contours, 0, (255, 255, 0), -1)

    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()
    cv2.imwrite(os.path.join(dataDir, 'Ground Truth', img['file_name']), I)
    print('Writing image: {c}/{t}'.format(c=c, t=len(imgIds)))

# for imgId in imgIds:
#     img = coco.loadImgs(imgId)[0]
#     I = io.imread(os.path.join(dataDir, img['file_name']))
