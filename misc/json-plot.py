import os
import json
import plotly
import plotly.graph_objs as go


file1 = '/home/vegas/data/Detectron_Output/e2e_faster_R-50-C4/RESULTS_Finetune-Faster-R-CNN_ResNet50-C4_CBIS_DDSM_val_400000.json'

with open(file1) as fp:
    data1 = json.load(fp)

iters1 = data1.keys()
afroc = []
iters = []
for iteration in iters1:
    iters.append(int(iteration))
    afroc.append(data1[iteration]["CBIS_DDSM_val"]["box"]["AFROC"])
pass

trace0 = go.Scatter(x=iters,
                    y=afroc,
                    name='Afroc')
data = [trace0]
layout = dict(title='Afroc Faster-R-CNN_ResNet50-C4 lr: {lr}'.format(lr='10^-5'),
              xaxis=dict(title='Iteration'),
              yaxis=dict(title='Afroc Score'))
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='/home/vegas/data/Detectron_Output/e2e_faster_R-50-C4/Afroc_Faster-R-CNN_ResNet50-C4_lr:10-5.html')