import os
import numpy as np


file = '/mnt/genesis/vegas/Detectron_Output/e2e_retinanet_R-101-FPN/test/CBIS_DDSM_val/retinanet/froc_retinanet_CBIS_DDSM_val.npy'

llf, nlf, fpr, thresh = np.load(file)

for index, item in enumerate(nlf):
    if item <= 11:
        nlf_f = nlf[index]
        llf_f = llf[index]
        thresh_f = thresh[index]
    else:
        break
print('llf: {llf_f}'.format(llf_f=llf_f))
print('nlf: {nlf_f}'.format(nlf_f=nlf_f))
print('threshold: {thresh_f}'.format(thresh_f=thresh_f))
