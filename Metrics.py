import os
import sys

import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

import torch

# -- IMPORT CUSTOM PATHS  ----------------------------------------------------------- #

# IMPORTS PATH TO THE PROJECT
current_project_path = os.path.dirname(os.path.realpath(__file__))
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_project_path)


from DataTransforms import PlytoNp
from Utils import compute_best_threshold

def compute_metrics(_label, _pred):

    f1_score = metrics.f1_score(_label, _pred)
    precision = metrics.precision_score(_label, _pred)
    recall = metrics.recall_score(_label, _pred)
    tn, fp, fn, tp = metrics.confusion_matrix(_label, _pred).ravel()

    return _pred, f1_score, precision, recall, (tn, fp, fn, tp)


def validation_metrics(_label, _pred):

    pred = _pred.cpu().numpy()
    label = _label.cpu().numpy().astype(int)
    trshld = compute_best_threshold(pred, label)
    pred = np.where(pred > trshld, 1, 0).astype(int)

    f1_score_list = []
    precision_list = []
    recall_list =  []
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    if pred.ndim == 2:
        batch_size = np.size(pred, 0)
        for i in range(batch_size):

            tmp_labl = label[i]
            tmp_pred = pred[i]

            precision_list.append(metrics.precision_score(tmp_labl, tmp_pred))
            recall_list.append(metrics.recall_score(tmp_labl, tmp_pred))
            f1_score_list.append(metrics.f1_score(tmp_labl, tmp_pred, average='binary'))
            tn, fp, fn, tp = metrics.confusion_matrix(tmp_labl, tmp_pred, labels=[0,1]).ravel()

            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            tp_list.append(tp)


        avg_f1_score = np.mean(np.array(f1_score_list))
        avg_precision = np.mean(np.array(precision_list))
        avg_recall = np.mean(np.array(recall_list))
        avg_tn = np.mean(np.array(tn_list))
        avg_fp = np.mean(np.array(fp_list))
        avg_fn = np.mean(np.array(fn_list))
        avg_tp = np.mean(np.array(tp_list))

    elif pred.ndim == 1:
        avg_f1_score = metrics.f1_score(label, pred, average='binary')
        avg_precision = metrics.precision_score(label, pred, average='binary')
        avg_recall = metrics.recall_score(label, pred, average='binary')
        avg_tn, avg_fp, avg_fn, avg_tp = metrics.confusion_matrix(label, pred, labels=[0, 1]).ravel()   
    
    else:
        raise ValueError('Wrong dimensions for prediction array')


    return trshld, pred, avg_f1_score, avg_precision, avg_recall, (avg_tn, avg_fp, avg_fn, avg_tp)



if __name__ == '__main__':

    GT_DIR = os.path.abspath('/media/arvc/data/datasets/ARVC_GZF/test/ply/bin_class')
    PRED_DIR = os.path.abspath('/home/arvc/PycharmProjects/ARVC_FPS/pred_clouds/20.12.2022')

    gt_dataset = sorted(os.listdir(GT_DIR))
    pred_dataset = sorted(os.listdir(PRED_DIR))

    f1_score_list = []
    precision_list = []
    recall_list = []
    conf_m = []

    for i, _ in enumerate(tqdm(gt_dataset)):
        gt_cloud_path = os.path.join(GT_DIR, gt_dataset[i])
        pred_cloud_path = os.path.join(PRED_DIR, pred_dataset[i])

        gt_cloud = PlytoNp(gt_cloud_path)
        pred_cloud = PlytoNp(pred_cloud_path)

        my_metrics = compute_metrics(gt_cloud, pred_cloud)

        f1_score_list.append(my_metrics[1])
        precision_list.append(my_metrics[2])
        recall_list.append(my_metrics[3])
        conf_m.append(my_metrics[4])

    print(f'Mean Precision: {np.mean(np.array(precision_list)):.4f}')
    print(f'Mean Recall: {np.mean(np.array(recall_list)):.4f}')
    print(f'Mean Confusion Matrix: {np.mean(np.array(conf_m), axis=0)}')

    print("Plot Done!")
