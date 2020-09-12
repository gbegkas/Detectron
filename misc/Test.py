#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time
import json
import plotly
import plotly.graph_objs as go

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
from matplotlib import pyplot as plt

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        '--models_dir',
        dest='models_dir',
        help='Directory containing models to test',
        default=None,
        type=str
    )
    parser.add_argument(
        '--first_model',
        dest='first_model',
        help='First model to start Testing',
        default=0,
        type=int
    )
    parser.add_argument(
        '--last_model',
        dest='last_model',
        help='Last model to Test',
        default=9999999,
        type=int
    )
    parser.add_argument(
        '--json_file',
        dest='json_file',
        help='Json file for resume testing. Leave empty for Testing from scratch',
        default=None,
        type=str
    )
    parser.add_argument(
        '--samples',
        dest='samples',
        help='Samples from checkpoints to test',
        default=1,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    dataset = cfg.TEST.DATASETS[0]

    backbone = None
    if 'ResNet50' in cfg.MODEL['CONV_BODY']:
        backbone = 'ResNet50'
    elif 'ResNet101' in cfg.MODEL['CONV_BODY'] and cfg.RESNETS['NUM_GROUPS'] == 1:
        backbone = 'ResNet101'
    elif 'ResNet101' in cfg.MODEL['CONV_BODY'] and cfg.RESNETS['NUM_GROUPS'] == 32:
        backbone = 'ResNeXt101'
    elif 'ResNet101' in cfg.MODEL['CONV_BODY'] and cfg.RESNETS['NUM_GROUPS'] == 64:
        backbone = 'ResNeXt101'
    elif 'VGG16' in cfg.MODEL['CONV_BODY']:
        backbone = 'VGG16'
    else:
        backbone = ''

    if cfg.FPN['FPN_ON']:
        backbone = backbone + '-FPN'
    else:
        backbone = backbone + '-C4'

    if cfg.MODEL['FASTER_RCNN'] and not cfg.MODEL['MASK_ON']:
        backbone = 'Faster-R-CNN_' + backbone
    elif cfg.MODEL['FASTER_RCNN'] and cfg.MODEL['MASK_ON']:
        backbone = 'Mask-R-CNN_' + backbone
    elif cfg.RETINANET['RETINANET_ON']:
        backbone = 'Retinanet_' + backbone

    if cfg.TRAIN['WEIGHTS']:
        backbone = 'Finetune-' + backbone
    else:
        backbone = 'From_scratch-' + backbone

    # if cfg.TEST['']

    print(backbone)

    auc_dict = {}
    model_list = {}
    res = {}
    last_model_from_json = 0
    if args.json_file:
        with open(args.json_file)as fp:
            temp = json.load(fp)

        res = {int(k): v for k, v in temp.items()}

        last_model_from_json = sorted(map(int, res.keys()))[len(res.keys())-1]
    f = False
    for path, _, files in os.walk(args.models_dir):
        for model in files:
            if 'model' in model and model[10:-4] not in auc_dict.keys():
                if 'final' not in model:
                    model_list[int(model[10:-4])] = os.path.join(path, model)
                else:
                    final = os.path.join(path, model)
                    f = True
    model_list_keys = sorted(model_list.keys())
    if f:
        model_list_keys.append(cfg.SOLVER['MAX_ITER'])
        model_list[cfg.SOLVER['MAX_ITER']] = final

    count = 0
    models = []
    afroc_scores = []
    for i in res.keys():
        afroc_scores.append(res[i][dataset]['box']['AFROC'])
        models.append(i)
    for model in model_list_keys:
        if args.last_model >= model >= args.first_model and model > last_model_from_json:
            count += 1
            if count % args.samples != 0:
                continue
            logger.info('Model to test:')
            logger.info(model)
            workspace.ResetWorkspace()
            results, auc_score, afroc_score = run_inference(
                model_list[model],
                ind_range=args.range,
                multi_gpu_testing=args.multi_gpu_testing,
                check_expected_results=True,
            )

            res[model] = results

            with open(os.path.join(cfg.OUTPUT_DIR,
                                   'RESULTS_{model}_{dataset}_{iter}.json'.format(model=backbone,
                                                                                  dataset=cfg.TEST.DATASETS[0],
                                                                                  iter=cfg.SOLVER['MAX_ITER'])),
                      'w') as fp:
                json.dump(res, fp, indent=1, separators=(',', ':'), sort_keys=True)
            models.append(model)
            afroc_scores.append(afroc_score)
            trace0 = go.Scatter(x=models,
                                y=afroc_scores,
                                name='Afroc')
            data = [trace0]
            layout = dict(title='Afroc {backbone} {dataset} lr: {lr}'.format(backbone=backbone,
                                                                             dataset=dataset,
                                                                             lr=cfg.SOLVER.BASE_LR),
                          xaxis=dict(title='Iteration'),
                          yaxis=dict(title='Afroc Score'))
            fig = dict(data=data, layout=layout)
            plotly.offline.plot(fig, filename=os.path.join(cfg.OUTPUT_DIR,
                                                           'Afroc_{backbone}_{dataset}_lr:{lr}.html'
                                                           .format(backbone=backbone,
                                                                   dataset=dataset,
                                                                   lr=cfg.SOLVER.BASE_LR)))

    # keys = map(int, res.keys())
    # keys = sorted(keys)
    # values = []
    # for key in keys:
    #     try:
    #         values.append(res[str(key)]['CBIS_DDSM_val']['box']['AUC'])
    #     except KeyError:
    #         values.append(res[key]['CBIS_DDSM_val']['box']['AUC'])
    #
    # fig = plt.figure()
    # plt.plot(keys, values)
    # plt.title('AUC Model {model} {dataset}'.format(model=backbone, dataset=cfg.TEST.DATASETS[0]))
    # plt.ylabel('Auc')
    # plt.xlabel('Iterations')
    # plt.axis([min(keys)-1000, max(keys)+1000, min(values)-0.05, max(values)+0.05])
    # fig.savefig(os.path.join(cfg.OUTPUT_DIR,
    #                      '{model}_{dataset}_{iter}.eps'.format(model=backbone,
    #                                                                dataset=cfg.TEST.DATASETS[0],
    #                                                                iter=cfg.SOLVER['MAX_ITER'])), format='eps')
    # fig.savefig(os.path.join(cfg.OUTPUT_DIR,
    #                      '{model}_{dataset}_{iter}.png'.format(model=backbone,
    #                                                                dataset=cfg.TEST.DATASETS[0],
    #                                                                iter=cfg.SOLVER['MAX_ITER'])))
    # plt.show()


