import os
import random
from shutil import copyfile


def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


# input_train = '/mnt/genesis/vegas/Databases/CBIS-DDSM-Original/Train'
# input_test = '/mnt/genesis/vegas/Databases/CBIS-DDSM-Original/Test'
#
# output_train = '/mnt/genesis/vegas/Databases/CBIS-DDSM/Train'
# output_validation = '/mnt/genesis/vegas/Databases/CBIS-DDSM/Validation'
# output_test = '/mnt/genesis/vegas/Databases/CBIS-DDSM/Test'

input_train = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM-Original/Train'
input_test = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM-Original/Test'

output = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM'
output_train = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM/Train'
output_validation = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM/Validation'
output_test = '/mnt/Cargo_2/Diploma_Thesis/Databases/CBIS-DDSM/Test'

mkdir(output)
mkdir(output_test)
mkdir(output_train)
mkdir(output_validation)
mkdir(os.path.join(output_train, 'annotations'))
mkdir(os.path.join(output_train, 'shapes'))
mkdir(os.path.join(output_test, 'annotations'))
mkdir(os.path.join(output_test, 'shapes'))
mkdir(os.path.join(output_validation, 'annotations'))
mkdir(os.path.join(output_validation, 'shapes'))

directories = [directory for _, directory, _ in os.walk(input_train)][0]
cases = list(set([directory.split(sep='_')[2] for directory in directories]))
random.shuffle(cases)

train_cases = cases[:int(.8*len(cases))]
validation_cases = cases[int(.8*len(cases)):]

train_directories = [directory for directory in directories for case in train_cases if case in directory]
validation_directories = [directory for directory in directories for case in validation_cases if case in directory]

for i, directory in enumerate(train_directories):
    print('Train: '+str(i+1)+'/'+str(len(train_directories)))
    for root, _, files in os.walk(os.path.join(input_train, directory)):
        for file in files:
            if '.png' in file:
                if 'mask' in file:
                    mask = os.path.join(root, file)
                    sep = mask.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(mask, os.path.join(output_train, 'annotations', image_id + '_mass.png'))
                else:
                    image = os.path.join(root, file)
                    sep = image.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(image, os.path.join(output_train, 'shapes', image_id + '.png'))

for i, directory in enumerate(validation_directories):
    print('Validation: ' + str(i+1) + '/' + str(len(validation_directories)))
    for root, _, files in os.walk(os.path.join(input_train, directory)):
        for file in files:
            if '.png' in file:
                if 'mask' in file:
                    mask = os.path.join(root, file)
                    sep = mask.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(mask, os.path.join(output_validation, 'annotations', image_id + '_mass.png'))
                else:
                    image = os.path.join(root, file)
                    sep = image.split(os.sep)
                    image_id = sep[len(sep) - 2]
                    copyfile(image, os.path.join(output_validation, 'shapes', image_id + '.png'))

for root, _, files in os.walk(input_test):
    for file in files:
        if '.png' in file:
            if 'mask' in file:
                mask = os.path.join(root, file)
                sep = mask.split(os.sep)
                image_id = sep[len(sep) - 2]
                copyfile(mask, os.path.join(output_test, 'annotations', image_id + '_mass.png'))
            else:
                image = os.path.join(root, file)
                sep = image.split(os.sep)
                image_id = sep[len(sep) - 2]
                copyfile(image, os.path.join(output_test, 'shapes', image_id + '.png'))
