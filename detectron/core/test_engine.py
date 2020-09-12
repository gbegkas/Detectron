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

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object
from detectron.utils.timer import Timer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils
import detectron.utils.metrics as metrics
import detectron.utils.plot as plot
import detectron.utils.save as save

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        child_func = generate_rpn_on_range
        parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
    weights_file, ind_range=None,
    multi_gpu_testing=False, gpu_id=0,
    check_expected_results=False,
):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = get_output_dir(dataset_name, training=False)
                results, auc_score, afroc_score = parent_func(
                    weights_file,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results, auc_score, afroc_score
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = get_output_dir(dataset_name, training=False)
            return child_func(
                weights_file,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results, auc_score, afroc_score = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results, auc, afroc_score


def test_net_on_dataset(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    multi_gpu=False,
    gpu_id=0
):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    model = ''
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            weights_file, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps, model = test_net(
            weights_file, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )

    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir
    )

    roc_data = metrics.calculate_roc(all_boxes, dataset, cfg.TEST.IOU)
    froc_data = metrics.calculate_froc(all_boxes, dataset, cfg.TEST.IOU)
    auc_score = {dataset.name: {u'box': {u'AUC': auc(roc_data[0], roc_data[1])}}}
    afroc_score = np.trapz(froc_data[0], froc_data[2])
    afroc = {dataset.name: {u'box': {u'AFROC': afroc_score}}}
    print('Afroc score: {:.4f}'.format(afroc_score))

    plot.plot_roc(roc_data, auc_score[dataset.name][u'box'][u'AUC'], dataset, model, output_dir)
    plot.plot_froc(froc_data, dataset, model, output_dir)
    plot.plot_afroc(froc_data, dataset, model, output_dir)

    save.np_save(np.stack(roc_data), 'roc', dataset, model, output_dir)
    save.np_save(np.stack(froc_data), 'froc', dataset, model, output_dir)

    results[dataset_name][u'box'].update(auc_score[dataset.name][u'box'])
    results[dataset_name][u'box'].update(afroc[dataset.name][u'box'])
    return results, auc_score, afroc_score


def multi_gpu_test_net_on_dataset(
    weights_file, dataset_name, proposal_file, num_images, output_dir
):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    opts += ['TEST.WEIGHTS', weights_file]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir, opts
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(
    weights_file,
    dataset_name,
    proposal_file,
    output_dir,
    ind_range=None,
    gpu_id=0
):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(weights_file, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None

        im = cv2.imread(entry['image'])
        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                model, im, box_proposals, timers
            )

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, misc_time, eta
                )
            )
            # logger.info(
            #     (
            #         'im_detect: range [{:d}, {:d}] of {:d}: '
            #         '{:d}/{:d} resize + RLE: {:.3f}s + {:.3f}s '
            #     ).format(
            #         start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
            #         start_ind + num_images, timers['maskResize'].average_time, timers['maskRLE'].average_time
            #     )
            # )

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.vis_one_image(
                im[:, :, ::-1],
                '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'),
                cls_boxes_i,
                segms=cls_segms_i,
                keypoints=cls_keyps_i,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes, all_segms, all_keyps, model.name


def initialize_model_from_cfg(weights_file, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils.initialize_gpu_from_weights_file(
        model, weights_file, gpu_id=gpu_id,
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]


"""
def get_iou(bb1, bb2):

    box1 = [int(round(i)) for i in bb1]
    box2 = [int(round(i)) for i in bb2]

    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max((x_right - x_left, 0)) * max((y_bottom - y_top), 0)

    # compute the area of both AABBs
    bb1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    bb2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def calculate_auc(all_boxes, dataset, weights_file, output_dir, gpu_id=0):
    y_true = []
    scores = []
    tpbs = []
    fpbs = []
    gt = []
    num_of_tumors = 0
    roidb = dataset.get_roidb()
    model = initialize_model_from_cfg(weights_file, gpu_id=gpu_id)

    for i, boxes in enumerate(all_boxes[1]):
        temp_tpbs = []
        temp_fpbs = []
        image_gt = []
        num_of_tumors += len(roidb[i]['gt_classes'])
        for box in boxes:
            scores.append(box[4])
            y_true_temp = 0
            for roi in roidb[i][u'boxes']:
                if get_iou(box[:-1], roi) >= 0.5:
                    image_gt.append(box[4])
                    y_true_temp = 1
                    break
            y_true.append(y_true_temp)
            if y_true_temp == 0:
                temp_fpbs.append(box[4])
        gt.append(image_gt)

        tpbs.append(temp_tpbs)
        fpbs.append(temp_fpbs)

    unlisted_FPs = [item for sublist in fpbs for item in sublist]
    unlisted_TPs = [item for sublist in tpbs for item in sublist]

    total_FPs, total_TPs, detected_gt = [], [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    c = 0
    for Thresh in all_probs[1::int(20/len(all_probs))]:
        t = []
        for image in gt:
            for pb in image:
                if pb >= Thresh:
                    t.append(1)
                    break
        c += 1

        detected_gt.append(sum(t))
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        # total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
        if c % 10 == 0:
            print(str(c) + '/' + str(int(len(all_probs)/1000) - 1))
    total_FPs.append(0)
    total_TPs.append(0)
    detected_gt.append(0)

    temp_FPs = total_FPs
    total_FPs = np.asarray(total_FPs) / float(len(all_boxes[1]))
    total_sensitivity = np.asarray(detected_gt) / float(num_of_tumors)
    fpr = np.asarray(temp_FPs) / float(len(unlisted_FPs))

    afroc_auc = np.trapz(total_sensitivity[::-1], fpr[::-1])

    fig = plt.figure()
    plt.xlabel('FPI', fontsize=12)
    plt.ylabel('TPR', fontsize=12)

    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')
    plt.ylim([0, 1])
    plt.savefig(os.path.join(output_dir,
                             'froc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.pdf'))

    fig = plt.figure()
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)

    fig.suptitle('Alternative free response receiver operating characteristic curve', fontsize=12)

    plt.plot(fpr, total_sensitivity, '-', color='#000000')
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir,
                             'afroc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.pdf'))

    nfp, nsens, nthresh = [], [], []
    for i, f in enumerate(total_FPs):
        if f <= 20:
            nfp.append(f)
            nsens.append(total_sensitivity[i])

    fig = plt.figure()
    plt.xlabel('FPI', fontsize=12)
    plt.ylabel('TPR', fontsize=12)

    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(nfp, nsens, '-', color='#000000')
    plt.ylim([0, 1])
    np.save(os.path.join(output_dir, 'froc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name)),
            [total_FPs, total_sensitivity, all_probs])

    plt.savefig(os.path.join(output_dir,
                             'froc_curve_20_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.pdf'))

    if max(y_true) == 0:
        auc = -1
        auc_score = {dataset.name: {u'box': {u'AUC': auc}}}
        print('Auc score:{:.4f}'.format(auc))
        return auc_score
    neg = y_true.count(0)
    fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(scores))
    auc = roc_auc_score(np.array(y_true), np.array(scores))
    auc_score = {dataset.name: {u'box': {u'AUC': auc}}}
    fpi = [neg/len(all_boxes)*x for x in fpr]

    print('Auc score:{:.4f}'.format(auc))
    print('Afroc Auc: {:.4f}'.format(afroc_auc))
    np.save('{out}/{model}_{dataset}_roc'.format(out=output_dir, model=model.name, dataset=dataset.name),
            [fpr, tpr, thresholds])
    np.save('{out}/{model}_{dataset}_froc'.format(out=output_dir, model=model.name, dataset=dataset.name),
            [fpi, tpr, thresholds])
    for i, entry in enumerate(tpr):
        if entry == 1:
            f = fpr[i]
            t = thresholds[i]
            break

    print('fpr: {f}'.format(f=f))
    print('threshold: {t}'.format(t=t))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of Model: {backbone} in {dataset} Dataset'
              .format(backbone=model.name, dataset=dataset.name))
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(output_dir, 'roc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.eps'),
        format='eps')
    plt.savefig(
        os.path.join(output_dir, 'roc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.png'))

    plt.figure()
    lw = 2
    plt.plot(fpi, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 10.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positives per Image')
    plt.ylabel('True Positive Rate')
    plt.title('FROC curve of Model: {backbone} in {dataset} Dataset'
              .format(backbone=model.name, dataset=dataset.name))
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(output_dir, 'froc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.eps'),
        format='eps')
    plt.savefig(
        os.path.join(output_dir, 'froc_curve_{back}_{dataset}'.format(back=model.name, dataset=dataset.name) + '.png'))
    return auc_score
"""
