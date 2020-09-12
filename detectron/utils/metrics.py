import numpy as np
from sklearn.metrics import roc_curve


def calculate_roc(all_boxes, dataset, iou_thresh=0.5):
    data = find_tp_fp(all_boxes, dataset, iou_thresh)
    return roc_curve(data[0], data[1])


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


def calculate_froc(all_boxes, dataset, iou_thresh=0.5):
    roidb = dataset.get_roidb()
    _, _, tp_pbs, fp_pbs, num_of_tumors, gt = find_tp_fp(all_boxes, dataset, iou_thresh)
    detected_rois = []
    detected_gt = []
    for i, image in enumerate(roidb):
        for roi in image[u'boxes']:
            max_box_score = 0
            detections = []
            for box in all_boxes[1][i]:
                if get_iou(box[:-1], roi) >= iou_thresh:
                    detections.append(box[4])
            detected_gt.append(detections)
            # for detection in detections:
            #     if detection[4] >= max_box_score:
            #         max_box_score = detection[4]
            # if max_box_score > 0:
            #     detected_rois.append(max_box_score)

    all_probs = sorted(set(tp_pbs + fp_pbs))
    total_fps, detected_gts = [], []
    c = 0
    # samples = len(all_probs)
    samples = 100
    flag = 0
    thresholds = [thresh for thresh in all_probs[1::int(len(all_probs)/samples)]]
    for threshold in thresholds:
        t = []
        for det in detected_gt:
            if len(det) == 0:
                continue
            for score in det:
                if score >= threshold:
                    t.append(1)
                    flag = 1
                    break
            # if flag:
            #     break
        c += 1
        detected_gts.append(sum(t))
        total_fps.append((np.asarray(fp_pbs) >= threshold).sum())

        # print(str(c) + '/' + str(len(thresholds)))

    total_fps.append(0)
    detected_gts.append(0)
    thresholds.append(0.0)

    llf = np.asarray(detected_gts) / float(num_of_tumors)
    nlf = np.asarray(total_fps) / float(len(all_boxes[1]))
    fpr = np.asarray(total_fps) / float(len(fp_pbs))
    thresholds = np.array(thresholds)

    return llf[::-1], nlf[::-1], fpr[::-1], thresholds[::-1]


def find_tp_fp(all_boxes, dataset, iou_thresh=0.5):
    roidb = dataset.get_roidb()
    y_true = []
    scores = []
    num_of_tumors = 0
    tp_pbs = []
    fp_pbs = []
    gt = []

    for i, boxes in enumerate(all_boxes[1]):
        num_of_tumors += len(roidb[i]['gt_classes'])
        for box in boxes:
            # box[4] is the detection score
            scores.append(box[4])
            y_true_temp = 0
            for roi in roidb[i][u'boxes']:
                if get_iou(box[:-1], roi) >= iou_thresh:
                    y_true_temp = 1
                    tp_pbs.append(box[4])
                    break

            y_true.append(y_true_temp)
            if y_true_temp == 0:
                fp_pbs.append(box[4])

    return y_true, scores, tp_pbs, fp_pbs, num_of_tumors, gt
