import numpy as np
from sklearn import metrics

class bcolors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def compute_best_threshold(_pred, _gt, _method = "pr"):
    trshld_per_cloud = []

    if _pred.ndim == 1:
        if _method == "roc":
            fpr, tpr, thresholds = metrics.roc_curve(_gt, _pred)
            gmeans = np.sqrt(tpr * (1 - fpr))
            index = np.argmax(gmeans)
            trshld_per_cloud.append(thresholds[index])

        elif _method == "pr":
            precision_, recall_, thresholds = metrics.precision_recall_curve(_gt, _pred)
            f1_score_ = (2 * precision_ * recall_) / (precision_ + recall_)
            index = np.argmax(f1_score_)
            trshld_per_cloud.append(thresholds[index])

        elif _method == "tuning":
            thresholds = np.arange(0.0, 1.0, 0.0001)
            f1_score_ = np.zeros(shape=(len(thresholds)))
            for index, elem in enumerate(thresholds):
                prediction_ = np.where(_pred > elem, 1, 0).astype(int)
                f1_score_[index] = metrics.f1_score(_gt, prediction_)

            index = np.argmax(f1_score_)
            trshld_per_cloud.append(thresholds[index])
        else:
            print('Error in the name of the method to use for compute best threshold')
    
    else:
        for cloud in range(len(_pred)):
            if _method == "roc":
                fpr, tpr, thresholds = metrics.roc_curve(_gt[cloud], _pred[cloud])
                gmeans = np.sqrt(tpr * (1 - fpr))
                index = np.argmax(gmeans)
                trshld_per_cloud.append(thresholds[index])

            elif _method == "pr":
                precision_, recall_, thresholds = metrics.precision_recall_curve(_gt[cloud], _pred[cloud])
                f1_score_ = (2 * precision_ * recall_) / (precision_ + recall_)
                index = np.argmax(f1_score_)
                trshld_per_cloud.append(thresholds[index])

            elif _method == "tuning":
                thresholds = np.arange(0.0, 1.0, 0.0001)
                f1_score_ = np.zeros(shape=(len(thresholds)))
                for index, elem in enumerate(thresholds):
                    prediction_ = np.where(_pred[cloud] > elem, 1, 0).astype(int)
                    f1_score_[index] = metrics.f1_score(_gt[cloud], prediction_)

                index = np.argmax(f1_score_)
                trshld_per_cloud.append(thresholds[index])
            else:
                print('Error in the name of the method to use for compute best threshold')
    

    return sum(trshld_per_cloud)/len(trshld_per_cloud)