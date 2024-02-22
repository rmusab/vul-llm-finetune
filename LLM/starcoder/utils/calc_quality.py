from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    roc_auc_score, precision_recall_curve, roc_curve
import numpy as np

"""
Functions for quality report for binary classification.
Function assume that there are two classes. In each function we pass probabilities of the positive class and true labels

Supports metrics: roc_auc, f1, macro_f1, macro_recall
roc_auc is a usual roc_auc
f1 is f1 for the positive class
macro_f1 is the macro average of f1 for 2 classes
macro_recall is the macro average of recall for 2 classes

How it is used: 
You pass validation probabilities and ground truth, and test probabilities and ground truth to the calc_quality function.
First, it computes an optimal thresholds for each metric on the validation data, and then applies it to the test data
For roc_auc there is no threshold, so we take a value of 0.5

You can also calculate validation quality with this function - just pass validation probabilities and ground truth as test data
"""
def find_best_threshold(probs, gt, metric='f1'):
    if metric in ['f1', 'macro_f1']:
        precision, recall, thresholds = precision_recall_curve(gt, probs)
        f1 = 2.0 * precision * recall / (precision + recall + 0.0000000000001)

        if metric == 'f1':
            arr_to_analyze = f1
        elif metric == 'macro_f1':
            precision2, recall2, thresholds2 = precision_recall_curve(1 - gt, 1.0 - probs)
            f1_2 = 2.0 * precision2 * recall2 / (precision2 + recall2 + 0.0000000000001)
            # make f1_2 the same shape as f1
            thresh_2_to_f1_2 = {thresholds2[i]: f1_2[i] for i in range(thresholds2.shape[0])}
            threshes_1_to_2 = 1.0 - thresholds
            f1_2_same_shape = np.array([thresh_2_to_f1_2[threshes_1_to_2[i]] for i in range(f1.shape[0] - 1)
                                        if threshes_1_to_2[i] in thresh_2_to_f1_2])
            f1_same_shape = np.array([f1[i] for i in range(f1.shape[0] - 1)
                                      if threshes_1_to_2[i] in thresh_2_to_f1_2])
            macro_f1 = (f1_same_shape + f1_2_same_shape) / 2.0
            arr_to_analyze = macro_f1
        else:
            raise NotImplementedError(metric)

    elif metric in ['macro_recall']:
        fpr, tpr, thresholds = roc_curve(gt, probs)
        arr_to_analyze = (tpr + (1.0 - fpr)) / 2.0
    else:
        raise NotImplementedError(metric)

    best_ind = np.argmax(arr_to_analyze)
    mest_metric_val = arr_to_analyze[best_ind]
    return mest_metric_val, thresholds[best_ind]

"""
The first return structure contains results per class for each threshold
The second return structure contains macro average of results for each threshold
The third is a best result of each metric
"""
def quality_full_report(probs_val, y_trues_val, probs, y_trues):
    res_per_class_for_metric = {}
    res_macro_for_metric = {}
    metric_vals_and_thresholds = {}
    roc_auc = roc_auc_score(y_trues, probs, average='macro')
    metric_vals_and_thresholds['roc_auc'] = {'best_val': roc_auc, 'best_thresh': 0.5}
    test_best_vals = {"roc_auc": roc_auc}
    for metric in ['f1', 'macro_f1', 'macro_recall']:
        best_val, best_thresh = find_best_threshold(probs_val, y_trues_val, metric=metric)
        metric_vals_and_thresholds[metric] = {'best_val': best_val, 'best_thresh': best_thresh}
        y_preds = probs > best_thresh
        precision12, recall12, f1_12, support_12 = precision_recall_fscore_support(y_trues, y_preds, labels=None,
                                                                                   average=None)
        res_per_class_for_metric[metric] = {class_num: {"recall": float(recall12[class_num]),
                                                        "precision": float(precision12[class_num]),
                                                        "f1": float(f1_12[class_num]),
                                                        "threshold": best_thresh} for class_num in [0, 1]}
        precision, recall, f1, support = precision_recall_fscore_support(y_trues, y_preds, labels=None, average='macro')
        res_macro_for_metric[metric] = {"recall": float(recall),
                                        "precision": float(precision),
                                        "f1": float(f1),
                                        "threshold": best_thresh}
        if metric == 'f1':
            test_best_vals[metric] = float(f1_12[1])
        elif metric == 'macro_f1':
            test_best_vals[metric] = float(f1)
        elif metric == 'macro_recall':
            test_best_vals[metric] = float(recall)

        return res_per_class_for_metric, res_macro_for_metric, test_best_vals

"""Quality report, the same procedure as the previous,
but return only best values"""
def quality_short_report(probs_val, y_trues_val, probs, y_trues):
    return quality_full_report(probs_val, y_trues_val, probs, y_trues)[2]

"""More convenient functions for the validation quality"""
def quality_full_report_val(probs_val, y_trues_val):
    return quality_full_report(probs_val, y_trues_val, probs_val, y_trues_val)

"""Validation quality report with only best values dict"""
def quality_short_report_val(probs_val, y_trues_val):
    return quality_full_report_val(probs_val, y_trues_val)[2]