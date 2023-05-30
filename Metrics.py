import numpy as np
import os
import sklearn.metrics as metrics
from DataTransforms import PlytoNp
from tqdm import tqdm


def compute_metrics(_label, _pred):

    f1_score = metrics.f1_score(_label, _pred)
    precision = metrics.precision_score(_label, _pred)
    recall = metrics.recall_score(_label, _pred)
    tn, fp, fn, tp = metrics.confusion_matrix(_label, _pred).ravel()

    return _pred, f1_score, precision, recall, (tn, fp, fn, tp)


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
