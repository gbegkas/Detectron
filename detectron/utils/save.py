import numpy as np
import os


def np_save(array, name, dataset, model, output_dir):
    filename = os.path.join(output_dir, name + '_' + model + '_' + dataset.name)
    np.save(filename, array)
