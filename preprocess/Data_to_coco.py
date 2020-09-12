#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM/Test'
# ROOT_DIR = '/mnt/genesis/vegas/Databases/INbreast/Test'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "INbreast_test",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'mass',
        'supercategory': 'mass',
    }
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    c = 0
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            if 'mass' and 'mask' not in image_filename:
                # print(image_filename)
                print(str(c + 1) + '/' + str(sum([len(files) for _, _, files in os.walk(IMAGE_DIR)])))
                c += 1
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root_ann, _, ann in os.walk(ANNOTATION_DIR):
                    annotation_files = filter_for_annotations(root_ann, ann, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        # print(annotation_filename)
                        class_id = 0
                        if 'mass' or 'mask' in annotation_filename:
                            class_id = 1

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1
            else:
                print('No image files')
                continue

    with open(os.path.join(ROOT_DIR, 'instances_shape_test2019.json'), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
