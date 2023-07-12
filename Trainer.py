import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import random
import shutil
import math
import socket
import sys
from datetime import datetime
import Config

class Trainer():

    def __init__(self, config_obj_ ):
        self.config = config_obj_ 
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.dataloader = None
        self.device = None
        self.data = None
        self.print_info = False
        self.print_every_x_batches = 10



    def apply_config():
        self.model = self.config.model

    def train(self):
        current_clouds = 0

        # TRAINING
        print('# ' + ('-' * 50))
        print('# TRAINING')
        self.model.train()
        for batch, self.data in enumerate(self.dataloader):
            
            features = self.data[0].to(self.device, dtype=torch.float32)
            labels = self.data[1].to(self.device, dtype=torch.int32)

            if len(self.data) == 3:
                filenames = self.data[2]  

            out = self.model(features)

            pred_conv = out[0]
            m = torch.nn.Sigmoid()
            pred_prob = m(pred_conv)

            loss = self.loss_fn(pred_prob, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_clouds += features.size(dim=0)

            if self.print_info:
                if batch % self.print_every_x_batches == 0 or batch == 0:  # print every (% X) batches
                    print(f' - [Batch: {current_clouds}/{len(self.dataloader.dataset)}],'
                          f' / Train Loss: {loss:.4f}')

    def valid(self):

        # VALIDATION
        print('-' * 50)
        print('VALIDATION')
        print('-'*50)
        model_.eval()
        f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, trshld_lst = [], [], [], [], [], []
        current_clouds = 0

        with torch.no_grad():
            for batch, (data, label, _) in enumerate(dataloader_):
                data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.float32)
                pred, m3x3, m64x64 = model_(data.transpose(1, 2))
                m = torch.nn.Sigmoid()
                pred = m(pred)

                avg_loss = loss_fn_(pred, label)
                loss_lst.append(avg_loss.item())

                trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, pred)
                trshld_lst.append(trshld)
                f1_lst.append(avg_f1)
                pre_lst.append(avg_pre)
                rec_lst.append(avg_rec)
                conf_m_lst.append(conf_m)

                current_clouds += data.size(0)

                if batch % 10 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
                    print(f'[Batch: {current_clouds}/{len(dataloader_.dataset)}]'
                        f'  [Avg Loss: {avg_loss:.4f}]'
                        f'  [Avg F1 score: {avg_f1:.4f}]'
                        f'  [Avg Precision score: {avg_pre:.4f}]'
                        f'  [Avg Recall score: {avg_rec:.4f}]')

        return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst, trshld_lst


    def compute_best_threshold(self, pred_, gt_):
        trshld_per_cloud = []
        method_ = self.config.train.threshold_method
        for cloud in range(len(pred_)):
            if method_ == "roc":
                fpr, tpr, thresholds = metrics.roc_curve(gt_[cloud], pred_[cloud])
                gmeans = np.sqrt(tpr * (1 - fpr))
                index = np.argmax(gmeans)
                trshld_per_cloud.append(thresholds[index])

            elif method_ == "pr":
                precision_, recall_, thresholds = metrics.precision_recall_curve(gt_[cloud], pred_[cloud])
                f1_score_ = (2 * precision_ * recall_) / (precision_ + recall_)
                index = np.argmax(f1_score_)
                trshld_per_cloud.append(thresholds[index])

            elif method_ == "tuning":
                thresholds = np.arange(0.0, 1.0, 0.0001)
                f1_score_ = np.zeros(shape=(len(thresholds)))
                for index, elem in enumerate(thresholds):
                    prediction_ = np.where(pred_[cloud] > elem, 1, 0).astype(int)
                    f1_score_[index] = metrics.f1_score(gt_[cloud], prediction_)

                index = np.argmax(f1_score_)
                trshld_per_cloud.append(thresholds[index])
            else:
                print('Error in the name of the method to use for compute best threshold')

        return sum(trshld_per_cloud)/len(trshld_per_cloud)
    
