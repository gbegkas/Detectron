import plotly.plotly as py
import plotly.graph_objs as go
import plotly

import json

file = '/mnt/genesis/vegas/Detectron_Output/e2e_faster_R-50-C4/RESULTS_Finetune-Faster-R-CNN_ResNet50-C4_CBIS_DDSM_val_200000.json'

with open(file) as fp:
    data = json.load(fp)

iteration = []
afroc = []
for key in data.keys():
    iteration.append(int(key))
    afroc.append(data[key]['CBIS_DDSM_val']['box']['AFROC'])

trace0 = go.Scatter(x=iteration,
                    y=afroc,
                    name='Afrco')
trace1 = go.Scatter(x=[0, 1],
                    y=[0, 1],
                    line=dict(
                        dash='dash'))
data = [trace1]
layout = dict(title='Afroc',
              xaxis=dict(title='Iteration'),
              yaxis=dict(title='Afroc Score'))
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='simple-connectgaps.html')

pass