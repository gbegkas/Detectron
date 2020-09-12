import os
from shutil import copyfile
from shutil import move
from random import randint

source = "/home/vegas/CBIS-DDSM"
destination = "/home/vegas/CBIS-DDSM-COCO_format"
train = os.path.join(destination, 'train')
test = os.path.join(destination, 'test')
val = os.path.join(destination, 'validation')

os.mkdir(destination)
os.mkdir(train)
os.mkdir(test)
os.mkdir(val)
os.mkdir(os.path.join(train, 'annotations'))
os.mkdir(os.path.join(train, 'shapes'))
os.mkdir(os.path.join(test, 'annotations'))
os.mkdir(os.path.join(test, 'shapes'))
os.mkdir(os.path.join(val, 'annotations'))
os.mkdir(os.path.join(val, 'shapes'))

counter = 0

for root, _, files in os.walk(source):
    if 'Train' in root:
        counter += 1
        for file in files:
            if '.png' in file:
                if 'mask' in file:
                    mask = os.path.join(root, file)
                    sep = mask.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(mask, os.path.join(train, 'annotations', image_id + '_mass.png'))
                else:
                    image = os.path.join(root, file)
                    sep = image.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(image, os.path.join(train, 'shapes', image_id + '.png'))

    elif 'Test' in root:
        counter += 1
        for file in files:
            if '.png' in file:
                if 'mask' in file:
                    mask = os.path.join(root, file)
                    sep = mask.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(mask, os.path.join(test, 'annotations', image_id + '_mass.png'))
                else:
                    image = os.path.join(root, file)
                    sep = image.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(image, os.path.join(test, 'shapes', image_id + '.png'))
    print('Processing {} of 1592'.format(counter))

validation = []
for root, _, files in os.walk(os.path.join(train, 'shapes')):
    for file in files:
        if randint(0,1) <= 0.2:
            validation.append(file[:-4])
            move(os.path.join(root, file), os.path.join(val, 'shapes', file))

for root, _, files in os.walk(os.path.join(train, 'annotations')):
    for file in files:
        if file[:-9] in validation:
            move(os.path.join(root, file), os.path.join(val, 'annotations', file))

