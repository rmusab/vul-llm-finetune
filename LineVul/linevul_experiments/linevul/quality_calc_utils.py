from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    roc_auc_score, precision_recall_curve, roc_curve
import numpy as np

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
            f1_2_same_shape = np.array([thresh_2_to_f1_2[threshes_1_to_2[i]] for i in range(f1.shape[0] - 1)])
            macro_f1 = (f1[:f1.shape[0] - 1] + f1_2_same_shape) / 2.0
            arr_to_analyze = macro_f1

    elif metric in ['macro_recall']:
        fpr, tpr, thresholds = roc_curve(gt, probs)
        arr_to_analyze = (tpr + (1.0 - fpr)) / 2.0

    best_ind = np.argmax(arr_to_analyze)
    mest_metric_val = arr_to_analyze[best_ind]
    return mest_metric_val, thresholds[best_ind]