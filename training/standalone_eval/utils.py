"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

MIT License

Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Copied from MMAction2
https://github.com/open-mmlab/mmaction2/blob/master/mmaction/core/evaluation/eval_detection.py
"""
import json
import numpy as np
from sklearn.metrics import precision_recall_curve


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    """ compute intersection-over-union along temporal axis for each pair of windows in pred_windows and gt_windows.
    Args:
        pred_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(
        0, np.minimum(pred_windows[:, 1], gt_windows[:, 1]) - np.maximum(pred_windows[:, 0], gt_windows[:, 0])
    )
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) \
            - np.minimum(pred_windows[:, 0], gt_windows[:, 0])  # not the correct union though
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def compute_temporal_iou_batch_cross(spans1, spans2):
    """
    Args:
        spans1: (N, 2) np.ndarray, each row defines a span [st, ed]
        spans2: (M, 2) np.ndarray, ...

    Returns:
        iou: (N, M) np.ndarray
        union: (N, M) np.ndarray
    >>> spans1 = np.array([[0, 0.2, 0.9], [0.5, 1.0, 0.2]])
    >>> spans2 = np.array([[0, 0.3], [0., 1.0]])
    >>> compute_temporal_iou_batch_cross(spans1, spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])  # (N, M)
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])  # (N, M)

    inter = np.clip(right - left, 0, None)  # (N, M)
    union = areas1[:, None] + areas2[None, :] - inter  # (N, M)

    iou = inter / union
    return iou, union


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', 't-end'
            and 'score'.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x['score'])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds)) # 用来标记每个预测是不是TP
    fp = np.zeros((num_thresholds, num_preds)) # 用来标记每个预测是不是FP

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction): #遍历"一个样本"的所有预测
        if pred['video-id'] in ground_truth_by_videoid:  # 如果这个预测的video-id在ground truth中
            gts = ground_truth_by_videoid[pred['video-id']] # 则把GT取出来
        else:
            fp[:, idx] = 1 # 如果不再groundtruth中, 则标记为FP
            continue

        _pred = np.array([[pred['t-start'], pred['t-end']], ]) # 取出预测的start和end
        _gt = np.array([[gt['t-start'], gt['t-end']] for gt in gts]) # 取出一个样本所有GT的start和end
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0] # 计算一个预测与所有GT的tIoU

        tiou_arr = tiou_arr.reshape(-1) # 把tIoU变成一维
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1] # 把tIoU排序, 从大到小, 用于后续AP的计算
        for t_idx, tiou_threshold in enumerate(tiou_thresholds): # 遍历所有tIoU阈值
            for j_idx in tiou_sorted_idx: # 遍历排序后的tIoU
                if tiou_arr[j_idx] < tiou_threshold: # 如果tIoU小于阈值, 则标记为FP
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0: # 
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])
    return ap

def compute_average_precision_detection_online(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10),
                                        ed_lengths=[1, 3, 5],
                                        max_len=10000):

    num_thresholds = len(tiou_thresholds)
    ap_results = {}

    for ed_length in ed_lengths:
        num_gts = len(ground_truth)
        num_preds = len(prediction)
        ap = np.zeros(num_thresholds)
        ap_zero = np.zeros(num_thresholds)
        ap_minus = np.zeros(num_thresholds)
        ap_pos = np.zeros(num_thresholds)

        if num_preds == 0:
            ap_results[ed_length] = ap_zero
            continue
 
        num_positive = float(num_gts)
        lock_gt = np.ones((num_thresholds, num_gts)) * -1
        # Sort predictions by decreasing score order.
        prediction.sort(key=lambda x: -x['score'])
        # Initialize true positive and false positive vectors.
        tp_zero = np.zeros((num_thresholds, num_preds))
        tp_pos = np.zeros((num_thresholds, num_preds))
        tp_minus = np.zeros((num_thresholds, num_preds))
        tp_true = np.zeros((num_thresholds, num_preds))
        fp = np.zeros((num_thresholds, num_preds))

        # Adaptation to query faster
        ground_truth_by_videoid = {}
        for i, item in enumerate(ground_truth):
            item['index'] = i
            ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

        # Assigning true positive to truly ground truth instances.
        for idx, pred in enumerate(prediction):
            if pred['video-id'] in ground_truth_by_videoid:
                gts = ground_truth_by_videoid[pred['video-id']]
            else:
                fp[:, idx] = 1
                continue

            _pred = np.array([[pred['t-start'], pred['t-end']], ])
            _gt = np.array([[gt['t-start'], gt['t-end']] for gt in gts])
            tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

            tiou_arr = tiou_arr.reshape(-1)
            # We would like to retrieve the predictions with highest tiou score.
            tiou_sorted_idx = tiou_arr.argsort()[::-1]
            for t_idx, tiou_threshold in enumerate(tiou_thresholds):
                for j_idx in tiou_sorted_idx:
                    if tiou_arr[j_idx] < tiou_threshold:
                        fp[t_idx, idx] = 1
                        break
                    if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                        continue

                    # Compute trunc_ed based on ed_length and max_len
                    gt_st = gts[j_idx]['t-start']
                    gt_ed = gts[j_idx]['t-end']
                    gen_time = pred['gentime']  # 从 prediction 中获取 gen_time
                    trunc_ed = min(gt_ed + ed_length, max_len)

                    # Compute weight based on gen_time
                    if gen_time <= gt_st:
                        weight_zero = 1
                        weight_minus = 1
                        weight_pos = 1
                    elif gt_st < gen_time <= gt_ed:
                        weight_zero = 1 - (gen_time - gt_st) / (gt_ed - gt_st) * 0.5
                        weight_minus = 1 - (gen_time - gt_st) / (gt_ed - gt_st) * 1
                        weight_pos = 1
                    elif gt_ed < gen_time <= trunc_ed:
                        weight_zero = 0.5 - (gen_time - gt_ed) / (trunc_ed - gt_ed) * 0.5
                        weight_minus = 0 - (gen_time - gt_ed) / (trunc_ed - gt_ed) * 1
                        weight_pos = 1 - (gen_time - gt_ed) / (trunc_ed - gt_ed) * 1
                    else:
                        weight_zero = 0
                        weight_minus = -1
                        weight_pos = 0

                    # Assign as true positive with weight
                    tp_zero[t_idx, idx] = weight_zero
                    tp_minus[t_idx, idx] = weight_minus
                    tp_pos[t_idx, idx] = weight_pos
                    tp_true[t_idx, idx] = 1
                    lock_gt[t_idx, gts[j_idx]['index']] = idx
                    break

                if fp[t_idx, idx] == 0 and tp_true[t_idx, idx] == 0:
                    fp[t_idx, idx] = 1

        # Compute cumulative sum of tp and fp
        tp_zero_cumsum = np.cumsum(tp_zero, axis=1).astype(float)
        tp_pos_cumsum = np.cumsum(tp_pos, axis=1).astype(float)
        tp_minus_cumsum = np.cumsum(tp_minus, axis=1).astype(float)
        tp_true_cumsum = np.cumsum(tp_true, axis=1).astype(float)
        fp_cumsum = np.cumsum(fp, axis=1).astype(float)

        recall_cumsum = tp_true_cumsum / num_positive
        precision_cumsum = tp_true_cumsum / (tp_true_cumsum + fp_cumsum)

        recall_zero_cumsum = tp_zero_cumsum / num_positive
        precision_zero_cumsum = tp_zero_cumsum / (tp_true_cumsum + fp_cumsum)

        recall_pos_cumsum = tp_pos_cumsum / num_positive
        precision_pos_cumsum = tp_pos_cumsum / (tp_true_cumsum + fp_cumsum)

        recall_minus_cumsum = tp_minus_cumsum / num_positive
        precision_minus_cumsum = tp_minus_cumsum / (tp_true_cumsum + fp_cumsum)

        # Compute AP for each tiou threshold using the interpolated precision-recall curve
        for t_idx in range(len(tiou_thresholds)):
            ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                      recall_cumsum[t_idx, :])
            ap_zero[t_idx] = interpolated_precision_recall(precision_zero_cumsum[t_idx, :],
                                                      recall_zero_cumsum[t_idx, :])
            ap_minus[t_idx] = interpolated_precision_recall(precision_minus_cumsum[t_idx, :],
                                                      recall_minus_cumsum[t_idx, :])
            ap_pos[t_idx] = interpolated_precision_recall(precision_pos_cumsum[t_idx, :],
                                                      recall_pos_cumsum[t_idx, :])

        # base_key = f"edlen_{ed_length}"
        # Store the AP results for this ed_length
        # ap_results[ed_length] = ap

        # 构造基础键
        base_key = f"edlen_{ed_length}"

        # 计算并赋值
        # ap_results[f"{base_key}"] = float(f"{np.mean(true_positive_mask) * 100:.2f}")
        ap_results[f"offline"] = ap
        ap_results[f"{base_key}_zero_online"] = ap_zero
        # ap_results[f"{base_key}_zero_online_gamma"] = np.nan_to_num(ap_zero / ap, nan=0)
        ap_results[f"{base_key}_minus_online"] = ap_minus
        ap_results[f"{base_key}_pos_online"] = ap_pos

    return ap_results  # 返回每个 ed_length 对应的 AP 结果


def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision

    ref: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py

    """
    # Check inputs
    assert len(y_true) == len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0  # True labels are all zeros
            # raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:  # Compute the interpolated precision
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:  # Compute the 11-point approximated AP
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:  # Compute the AP using precision at every additionally recalled sample
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])