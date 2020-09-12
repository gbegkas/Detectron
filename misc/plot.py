import os
import numpy as np
import json
from matplotlib import pyplot as plt
# from sklearn import metrics

MODELS = {
    'faster50c4': 'Faster R-CNN - ResNet 50 C4',
    'faster50fpn': 'Faster R-CNN - ResNet 50 FPN',
    'faster101c4': 'Faster R-CNN - ResNet 101 C4',
    'faster101fpn': 'Faster R-CNN - ResNet 101 FPN',
    'fastervgg': 'Faster R-CNN - VGG 16',
    'fasterx': 'Faster R-CNN - ResNeXt 101 32x8d FPN',
    'retinanet50': 'RetinaNet - ResNet 50 FPN',
    'retinanet101': 'RetinaNet - ResNet 101 FPN',
    'retinanet101noaug': 'RetinaNet - ResNet 101 FPN without augmentation'
}

DATABASES = {
    'cbis': 'CBIS-DDSM',
    'inbreast': 'INbreast'
}

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

directory = os.path.join(os.getcwd(), 'Results')
froc = {'CBIS-DDSM': [],
        'INbreast': []}
roc = {'CBIS-DDSM': [],
       'INbreast': []}

afroc = {'CBIS-DDSM': [],
         'INbreast': []}

cbis = []
inbreast = []

plt.rcParams['font.family'] = 'Linux Libertine O'
#
# for root, _, files in os.walk(directory):
#     split = os.path.split(root)
#     database = split[-1]
#     model = os.path.split(split[-2])[-1]
#     for file in files:
#         if '.json' and 'RESULTS' in file:
#             iter = []
#             afroc = []
#             with open(os.path.join(root, file), 'r') as fp:
#                 data = json.load(fp)
#
#             if 'CBIS' in file:
#                 for key in data.keys():
#                     iter.append(int(key))
#                     afroc.append(data[key]['CBIS_DDSM_val']['box']['AFROC'])
#                 cbis.append([MODELS[database], (iter, afroc)])
#             elif 'INbreast' in file:
#                 for key in data.keys():
#                     iter.append(int(key))
#                     afroc.append(data[key]['INbreast_test']['box']['AFROC'])
#                 inbreast.append([MODELS[database], (iter, afroc)])
# pass
#
# plt.rcParams['font.family'] = 'Linux Libertine O'
# plt.rcParams['font.size'] = 12
#
# plt.axes(xlim=(0, 200100), ylim=(0.7, 0.97))
# plt.xlabel('Iteration')
# plt.ylabel('AFROC')
#
# for data in cbis:
#     plt.axes(xlim=(0, 200100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'ResNet 50' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('r50val-cbis.pdf')
# plt.show()
# for data in cbis:
#     plt.axes(xlim=(0, 200100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'ResNet 101' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('r101val-cbis.pdf')
# plt.show()
#
# for data in cbis:
#     plt.axes(xlim=(0, 200100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'C4' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('C4val-cbis.pdf')
# plt.show()
#
# for data in cbis:
#     plt.axes(xlim=(0, 200100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'FPN' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('FPNval-cbis.pdf')
# plt.show()
#
# for data in cbis:
#     plt.axes(xlim=(0, 200100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'RetinaNet' in data[0] and 'augmentation' not in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('retinaval-cbis.pdf')
# plt.show()
#
# for data in cbis:
#     plt.axes(xlim=(0, 200100), ylim=(0.5, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'RetinaNet' in data[0] and '101' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('ret101val-cbis.pdf')
# plt.show()
#
# for data in inbreast:
#     plt.axes(xlim=(0, 100100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'ResNet 50' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('r50val-inbreast.pdf')
# plt.show()
# for data in inbreast:
#     plt.axes(xlim=(0, 100100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'ResNet 101' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('r101val-inbreast.pdf')
# plt.show()
#
# for data in inbreast:
#     plt.axes(xlim=(0, 100100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'C4' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('C4val-inbreast.pdf')
# plt.show()
#
# for data in inbreast:
#     plt.axes(xlim=(0, 100100), ylim=(0.7, 0.97))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'Faster' in data[0] and 'FPN' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('FPNval-inbreast.pdf')
# plt.show()
#
# for data in inbreast:
#     plt.axes(xlim=(0, 100100), ylim=(0.7, 0.99))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'RetinaNet' in data[0] and 'augmentation' not in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('retinaval-inbreast.pdf')
# plt.show()
#
# for data in inbreast:
#     plt.axes(xlim=(0, 100100), ylim=(0.5, 0.99))
#     plt.xlabel('Iteration')
#     plt.ylabel('AFROC')
#     if 'RetinaNet' in data[0] and '101' in data[0]:
#         plt.plot(data[1][0], data[1][1], label=data[0])
#
# plt.legend(loc='lower right')
# plt.savefig('ret101val-inbreast.pdf')
# plt.show()

# plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('val-cbis.pdf')
# plt.show()

# for root, _, files in os.walk(directory):
#     split = os.path.split(root)
#     database = split[-1]
#     model = os.path.split(split[-2])[-1]
#     for file in files:
#         if '.npy' in file:
#             if 'froc' in file:
#                 froc[DATABASES[database]].append([MODELS[model], np.load(os.path.join(root, file))])
#             else:
#                 roc[DATABASES[database]].append([MODELS[model], np.load(os.path.join(root, file))])
# pass
#
# plt.rcParams['font.family'] = 'Linux Libertine O'
# plt.rcParams['font.size'] = 12
#
# plt.axes(xlim=(0, 210), ylim=(0, 1.02))
# plt.xlabel('False Positive per Image')
# plt.ylabel('True Positive Rate')
#
# for data in froc['CBIS-DDSM']:
#     plt.plot(data[1][1], data[1][0], label=data[0])
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('froc-cbis.pdf')
# plt.show()
#
# plt.axes(xlim=(0, 350), ylim=(0, 1.02))
# plt.xlabel('False Positive per Image')
# plt.ylabel('True Positive Rate')
#
# for data in froc['INbreast']:
#     plt.plot(data[1][1], data[1][0], label=data[0])
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('froc-inbreast.pdf')
# plt.show()
#
#
# plt.axes(xlim=(0, 1), ylim=(0, 1.02))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
#
# for data in froc['CBIS-DDSM']:
#     plt.plot(data[1][2], data[1][0], label=data[0] + ' AUC: {0:.4f}'.format(metrics.auc(data[1][2], data[1][0])))
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('afroc-cbis.pdf')
# plt.show()
#
# plt.axes(xlim=(0, 1), ylim=(0, 1.02))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
#
# for data in froc['INbreast']:
#     plt.plot(data[1][2], data[1][0], label=data[0] + ' AUC: {0:.4f}'.format(metrics.auc(data[1][2], data[1][0])))
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('afroc-inbreast.pdf')
# plt.show()
#
#
# plt.axes(xlim=(0, 1), ylim=(0, 1.02))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
#
# for data in roc['CBIS-DDSM']:
#     plt.plot(data[1][0], data[1][1], label=data[0] + ' AUC: {0:.4f}'.format(metrics.auc(data[1][0], data[1][1])))
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('roc-cbis.pdf')
# plt.show()
#
# plt.axes(xlim=(0, 1), ylim=(0, 1.02))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
#
# for data in roc['INbreast']:
#     plt.plot(data[1][0], data[1][1], label=data[0] + ' AUC: {0:.4f}'.format(metrics.auc(data[1][0], data[1][1])))
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('roc-inbreast.pdf')
# plt.show()

# fig = plt.figure(1)
#
# plt.figure(figsize=(5.8, 5.5))
#
# plt.axes(xlim=(0, 1), ylim=(0, 1))
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('False Positive Rate', fontsize=15)
# plt.ylabel('True Positive Rate', fontsize=15)
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.savefig('rocspace.pdf')
# plt.show()
#
# fig = plt.figure(1)
#
# plt.figure(figsize=(5.8, 5.5))
#
# plt.axes(xlim=(-0.008, 1), ylim=(0, 1.008))
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('False Positive Rate', fontsize=15)
# plt.ylabel('True Positive Rate', fontsize=15)
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.plot([0, 0, 1], [0, 1, 1], lw=2, color='darkorange', label='ROC curve (area = %0.1f)' %np.trapz([0, 1, 1], [0, 0, 1]))
# plt.fill_between([0, 0, 1], [0, 1, 1], color=(0.4, 0.4, 0.4, 0.2))
# plt.legend(loc='lower right', fontsize=14)
# plt.savefig('perfroc.pdf')
# plt.show()
#
#
#
# data = np.load('/mnt/Cargo_2/Sync/Πανεπιστημιο/Diploma Thesis/Python Scripts/Results/retinanet101/inbreast/roc_retinanet_INbreast_test.npy')
#
# plt.figure(figsize=(5.8, 5.5))
#
# plt.axes(xlim=(0, 1), ylim=(0, 1.02))
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('False Positive Rate', fontsize=15)
# plt.ylabel('True Positive Rate', fontsize=15)
#
# plt.fill_between(data[0], data[1], color=(0.4, 0.4, 0.4, 0.2))
#
# plt.plot(data[0], data[1], color='darkorange', label='ROC curve (area = %0.4f)' %np.trapz(data[1], data[0]))
#
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#
# plt.legend(loc="lower right", fontsize=14)
# plt.savefig('rocexample.pdf')
# plt.show()

# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.figure(figsize=(5,5))
# plt.axis('off')
# plt.axes().set_aspect('equal')
# plt.plot([3,3,5,5,3], [4,6.1,6.1,4,4], lw=4)
# plt.plot([3.1,3.1,6,6,3.1], [3,6,6,3,3], lw=4)
# plt.savefig('piou.pdf')
# plt.show()
# #
# plt.figure(figsize=(5,5))
# plt.axis('off')
# plt.axes().set_aspect('equal')
# plt.plot([3,3,6,6,3], [3.05,6.05,6.05,3.05,3.05], lw=4)
# plt.plot([3.05,3.05,6.05,6.05,3.05], [3,6,6,3,3], lw=4)
# plt.savefig('eiou.pdf')
# plt.show()
#
# plt.figure(figsize=(5,5))
# plt.axis('off')
# plt.axes().set_aspect('equal')
# plt.plot([3,3,6,6,3], [3.2,6.2,6.2,3.2,3.2], lw=4)
# plt.plot([3.2,3.2,6.2,6.2,3.2], [3,6,6,3,3], lw=4)
# plt.savefig('giou.pdf')
# plt.show()

# plt.figure(figsize=(5,5))
# plt.axis('off')
# plt.axes().set_aspect('equal')
# plt.plot([3,3,5,5,3], [4,6,6,4,4], lw=4)
# plt.plot([4,4,6,6,4], [3,5,5,3,3], lw=4, c='#1f77b4')
# plt.fill([4,4,5,5,4], [4,5,5,4,4])
# plt.savefig('iou.pdf')
# plt.show()

# plt.figure(figsize=(5,5))
# plt.axis('off')
# plt.axes().set_aspect('equal')
# plt.plot([3,3,5,5,3], [4,6,6,4,4], lw=4)
# plt.plot([4,4,6,6,4], [3,5,5,3,3], lw=4, c='#1f77b4')
# plt.fill([3,3,5,5,6,6,4,4,3], [4,6,6,5,5,3,3,4,4])
# plt.savefig('area.pdf')
# plt.show()

# data = np.load('/mnt/Cargo_2/Sync/Πανεπιστημιο/Diploma Thesis/Python Scripts/Results/retinanet101/inbreast/froc_retinanet_INbreast_test.npy')
#
# plt.figure(figsize=(5.8, 5.5))
#
# plt.axes(xlim=(0, 140), ylim=(0, 1.02))
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('False Positive per Image', fontsize=15)
# plt.ylabel('True Positive Rate', fontsize=15)
#
# # plt.fill_between(data[1], data[0], color=(0.4, 0.4, 0.4, 0.2))
#
# plt.plot(data[1], data[0], color='darkorange', lw=2, label='ROC curve (area = %0.4f)' %np.trapz(data[0], data[1]))
#
# # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#
# # plt.legend(loc="lower right", fontsize=14)
# plt.savefig('frocexample.pdf')
# plt.show()
#
# plt.figure(figsize=(5.8, 5.5))
#
# plt.axes(xlim=(0, 1), ylim=(0, 1.02))
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('False Positive Rate', fontsize=15)
# plt.ylabel('True Positive Rate', fontsize=15)
#
# # plt.fill_between(data[1], data[0], color=(0.4, 0.4, 0.4, 0.2))
#
# plt.plot(data[2], data[0], color='darkorange', lw=2, label='ROC curve (area = %0.4f)' %np.trapz(data[0], data[2]))
#
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#
# plt.legend(loc="lower right", fontsize=14)
# plt.savefig('afrocexample.pdf')
# plt.show()

# afroc = []
#
# for root, _, files in os.walk(directory):
#     split = os.path.split(root)
#     model = split[-1]
#     # model = os.path.split(split[-2])[-1]
#     for file in files:
#         if 'noaug' in root:
#             continue
#         if '.json' in file:
#             if 'CBIS' in file:
#                 database = 'CBIS-DDSM'
#             elif 'INbreast' in file:
#                 database = 'INbreast'
#             if 'RESULTS' in file:
#                 with open(os.path.join(root, file), 'r') as f:
#                     afroc[database].append([MODELS[model], json.load(f)])
#
# pass
#
# afroc_c = []
# afroc_i = []
# # x = []
# # models = []
# # auc = []
#
# data = {}
#
# for dataset in afroc.keys():
#     models = []
#     for i, model in enumerate(afroc[dataset]):
#         name = afroc[database][i][0]
#         auc = []
#         x = []
#         for iter in afroc[dataset][i][1].keys():
#             x.append(int(iter))
#             if 'CBIS' in dataset:
#                 auc.append(afroc[dataset][i][1][iter]['CBIS_DDSM_val']['box']['AFROC'])
#             elif 'INbreast' in dataset:
#                 auc.append(afroc[dataset][i][1][iter]['INbreast_test']['box']['AFROC'])
#         models.append([name, (x, auc)])
#     data[dataset] = models
#
#
# plt.rcParams['font.family'] = 'Linux Libertine O'
# plt.rcParams['font.size'] = 12
#
# plt.axes(xlim=(0, 205000), ylim=(0, 1.02))
# plt.xlabel('Iterations')
# plt.ylabel('AFROC AUC')
#
# for d in data['CBIS-DDSM']:
#     plt.plot(d[1][0], d[1][1], label=d[0])
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('afroc-cbis-val.pdf')
# plt.show()
#
# plt.rcParams['font.family'] = 'Linux Libertine O'
# plt.rcParams['font.size'] = 12
# plt.axes(xlim=(0, 105000), ylim=(0, 1.02))
# plt.xlabel('Iterations')
# plt.ylabel('AFROC AUC')
# for d in data['INbreast']:
#     plt.plot(d[1][0], d[1][1], label=d[0])
#
# # plt.plot(data[1], data[0], label=file[:-4] + ' AUC: ' + "{0:.4f}".format(auc))
# # plt.margins(x=0.01)
# plt.legend(loc='lower right')
#
# plt.savefig('afroc-inbreast-val.pdf')
# plt.show()

models = ['Faster RCNN R50-C4',
          'Faster RCNN R50-FPN',
          'Faster RCNN R101-C4',
          'Faster RCNN R101-FPN',
          'Faster RCNN X101-32x8d',
          'Faster RCNN VGG16',
          'RetinaNet R50-FPN',
          'RetinaNet R101-FPN',
          'RetinaNet R101-FPN w/o augm']

nf = [0.7968,
      0.6787,
      0.7442,
      0.8025,
      0.7527,
      0.8288,
      0.9412,
      0.9254,
      0.8692]

f = [0.8689,
     0.9055,
     0.8741,
     0.9547,
     0.9082,
     0.8678,
     0.9646,
     0.9751,
     0.9006]

groups = len(models)
index = np.arange(groups)
bar_width = 0.35
opacity = 1.0

# plt.subplot(113)
rects1 = plt.barh(index, nf, bar_width, alpha=opacity, label='w/o finetune')
rects2 = plt.barh(index + bar_width, f, bar_width, alpha=opacity, label='w/ finetune')

plt.ylabel('Μοντέλο')
plt.xlabel('AFROC')

plt.yticks(index + bar_width, models)
plt.legend(bbox_to_anchor=(-0.45, 0.03), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.savefig('finetune.pdf')
plt.show()

