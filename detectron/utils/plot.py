import os
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly


def plot_roc(roc_data, auc, dataset, model, output_dir):
    trace0 = go.Scatter(x=roc_data[0],
                        y=roc_data[1],
                        name='ROC')
    trace1 = go.Scatter(x=[0, 1],
                        y=[0, 1],
                        line=dict(
                            dash='dash'))
    data = [trace0, trace1]
    layout = dict(title='ROC curve of Model: {backbone} in {dataset} Dataset, AUC: {auc:.4f}'
                  .format(backbone=model, dataset=dataset.name, auc=auc),
                  xaxis=dict(title='FPR'),
                  yaxis=dict(title='TPR'))
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=os.path.join(output_dir, 'roc_curve_{back}_{dataset}.html'
                                                   .format(back=model, dataset=dataset.name)))

    # plt.figure()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve of Model: {backbone} in {dataset} Dataset'.format(backbone=model, dataset=dataset.name))
    # lw = 2
    # plt.plot(roc_data[0], roc_data[1], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.4f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(output_dir, 'roc_curve_{back}_{dataset}'.format(back=model, dataset=dataset.name)+'.pdf')
    #             , format='pdf')


def plot_froc(froc_data, dataset, model, output_dir):
    trace0 = go.Scatter(x=froc_data[1],
                        y=froc_data[0],
                        name='FROC')
    data = [trace0]
    layout = dict(title='FROC curve of Model: {backbone} in {dataset} Dataset'
                  .format(backbone=model, dataset=dataset.name),
                  xaxis=dict(title='FPI'),
                  yaxis=dict(title='TPR'))
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=os.path.join(output_dir, 'froc_curve_{back}_{dataset}.html'
                                                   .format(back=model, dataset=dataset.name)))

    # plt.figure()
    # plt.xlabel('False Positives per Image')
    # plt.ylabel('True Positive Rate')
    # plt.plot(froc_data[1], froc_data[0], color='darkorange')
    # plt.title('FROC curve of Model: {backbone} in {dataset} Dataset'.format(backbone=model, dataset=dataset.name))
    # plt.ylim([0.0, 1.05])
    # plt.savefig(
    #     os.path.join(output_dir, 'froc_curve_{back}_{dataset}'.format(back=model, dataset=dataset.name) + '.pdf')
    #     , format='pdf')


def plot_afroc(afroc_data, dataset, model, output_dir):
    auc = np.trapz(afroc_data[0], afroc_data[2])

    trace0 = go.Scatter(x=afroc_data[2],
                        y=afroc_data[0],
                        name='AFROC')
    trace1 = go.Scatter(x=[0, 1],
                        y=[0, 1],
                        line=dict(
                            dash='dash'))
    data = [trace0, trace1]
    layout = dict(title='AFROC curve of Model: {backbone} in {dataset} Dataset, AUC: {auc:.4f}'
                  .format(backbone=model, dataset=dataset.name, auc=auc),
                  xaxis=dict(title='FPR'),
                  yaxis=dict(title='TPR'))
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=os.path.join(output_dir, 'afroc_curve_{back}_{dataset}.html'
                                                   .format(back=model, dataset=dataset.name)))

    # plt.figure()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('AFROC curve of Model: {backbone} in {dataset} Dataset'.format(backbone=model, dataset=dataset.name))
    # lw = 2
    # plt.plot(afroc_data[2], afroc_data[0], color='darkorange',
    #          lw=lw, label='AFROC curve (area = %0.4f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend(loc="lower right")
    # plt.savefig(
    #     os.path.join(output_dir, 'afroc_curve_{back}_{dataset}'.format(back=model, dataset=dataset.name) + '.pdf')
    #     , format='pdf')
