# Plot AUC over iterations

import os
import json
from utils.getListOfFiles import getListOfFiles
from matplotlib import pyplot as plt


in_dir = '/home/gru/RESULTS-detectron/Json/retinanet'
out_dict = {}

files = getListOfFiles(in_dir)

for file in files:
    if 'Best' not in os.path.basename(file):
        with open(file) as fp:
            data = json.load(fp)

        max_auc = 0
        max_ap50 = 0
        for key in sorted(data.keys()):
            if int(key) < 50000:
                if max_auc < data[key]['CBIS_DDSM_val']['box']['AUC']:
                    max_auc = data[key]['CBIS_DDSM_val']['box']['AUC']
                    max_key = key
                if max_ap50 < data[key]['CBIS_DDSM_val']['box']['AP50']:
                    max_ap50 = data[key]['CBIS_DDSM_val']['box']['AP50']
                    key_ap50 = key

        print('Iteration: {iter}'.format(iter=max_key))
        for metric in data[max_key]['CBIS_DDSM_val']['box'].keys():
            print('{metric}: {data}'.format(metric=metric, data=data[max_key]['CBIS_DDSM_val']['box'][metric]))

        keys = map(int, data.keys())
        keys = sorted(keys)
        values = []
        ap50 = []
        for key in keys:
            if data[str(key)]['CBIS_DDSM_val']['box']['AUC'] == -1:
                values.append(0)
            else:
                values.append(data[str(key)]['CBIS_DDSM_val']['box']['AUC'])
            ap50.append(data[str(key)]['CBIS_DDSM_val']['box']['AP50'])

        fig = plt.figure()
        plt.plot(keys, values, label='max auc = %d' % int(max_key))
        plt.title('AUC Model {model}'.format(model=os.path.basename(file)[:-5]))
        plt.ylabel('Auc')
        plt.xlabel('Iterations')
        plt.axis([min(keys) - 1000, max(keys) + 1000, min(values) - 0.01, max(values) + 0.01])
        plt.legend(loc="lower right")
        plt.show()
        fig.savefig(os.path.join(in_dir, 'AUC_' + os.path.basename(file)[:-5] + '.png'))

        fig = plt.figure()
        plt.plot(keys, ap50)
        plt.title('AP50 Model {model}'.format(model=os.path.basename(file)[:-5]))
        plt.ylabel('AP50')
        plt.xlabel('Iterations')
        plt.axis([min(keys) - 1000, max(keys) + 1000, min(ap50) - 0.01, max(ap50) + 0.01])
        fig.savefig(os.path.join(in_dir, 'AP50_' + os.path.basename(file)[:-5] + '.png'))
