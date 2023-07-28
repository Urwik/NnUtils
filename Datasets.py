import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from plyfile import PlyData

from tqdm import tqdm

class PLYDatasetPlaneCount(Dataset):
    def __init__(self, root_dir = 'my_dataset_dir', features=None, labels_file='plane_count.csv', normalize=False):
        super().__init__()
        self.root_dir = root_dir

        if features is None:
            self.features = [0, 1, 2]
        else:
            self.features = features

        self.labelsFile = labels_file
        self.normalize = normalize

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        filename = file.split('.')[0]
        path_to_file = os.path.join(self.root_dir, file)
        labels_path = os.path.join(self.root_dir, self.labelsFile)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = data[:, self.features]

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = data[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0, 1, 2]] = xyz

        # Get number of planes from csv file
        data_frame = pd.read_csv(labels_path, delimiter=',')
        labels_dict = data_frame.set_index('File')['Planes'].to_dict()
        label = labels_dict.get(int(filename))

        return features, label, filename


class PLYDataset(Dataset):
    def __init__(self,
                 _mode = 'train', 
                 _root_dir = 'my_dataset_dir',
                 _coord_idx = None,
                 _feat_idx = None,
                 _label_idx = None,
                 _normalize = False,
                 _binary = False,
                 _add_range = False, 
                 _compute_weights = False):
        
        super().__init__()
        self.mode = _mode
        self.root_dir = _root_dir
        self.coord_idx = _coord_idx if _coord_idx is not None else [0, 1, 2]
        self.feat_idx = _feat_idx if _feat_idx is not None else [0, 1, 2]
        self.label_idx = _label_idx if _label_idx is not None else [-1]
        self.normalize = _normalize
        self.binary = _binary
        self.add_range = _add_range
        self.compute_weights = _compute_weights
        self.weights = []
        self.coords = []
        self.features = []
        self.labels = []
        self.dataset_size = 0

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)

        if self.compute_weights:
            self.weights = []
            # COMPUTE WEIGHTS FOR EACH LABEL IN THE WHOLE DATASET
            print('-'*50)
            print("COMPUTING LABEL WEIGHTS")
            for file in tqdm(self.dataset):
                # READ THE FILE
                path_to_file = os.path.join(self.root_dir, file)
                ply = PlyData.read(path_to_file)
                data = ply["vertex"].data
                data = np.array(list(map(list, data)))

                # CONVERT TO BINARY LABELS
                labels = data[:, self.label_idx].copy()

                if self.binary:
                    labels[labels > 0] = 1

                labels = np.sort(labels, axis=None)
                k_lbl, weights = np.unique(labels, return_counts=True)
                # SI SOLO EXISTE UNA CLASE EN LA NUBE (SOLO SUELO)
                if k_lbl.size < 2:
                    if k_lbl[0] == 0:
                        weights = np.array([1, 0])
                    else:
                        weights = np.array([0, 1])
                else:
                    weights = weights / len(labels)

                if len(self.weights) == 0:
                    self.weights = weights
                else:
                    self.weights = np.vstack((self.weights, weights))

            self.weights = np.mean(self.weights, axis=0).astype(np.float32)

    def __len__(self):
        if self.dataset_size > 0:
            return self.dataset_size
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        path_to_file = os.path.join(self.root_dir, file)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        self.coords = data[:, self.coord_idx].copy()
        self.features = data[:, self.feat_idx].copy()
        self.labels = data[:, self.label_idx].copy()

        if self.feat_idx[:3] == [0, 1, 2]:
            if self.normalize:
                xyz = self.coords.copy()
                centroid = np.mean(xyz, axis=0)
                xyz -= centroid
                furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
                xyz /= furthest_distance
                self.features[:, [0, 1, 2]] = xyz

        if self.add_range:
            xyz = self.coords.copy()
            range = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            range = range[:, None]
            self.features = np.hstack((self.features, range))

        if self.binary:
            self.labels[self.labels > 0] = 1


        return self.coords, self.features, self.labels, self.dataset[index].split('.')[0]


class MinkDataset(Dataset):

    def __init__(self,
                 _mode='train',
                 _root_dir='my_dataset_dir',
                 _coord_idx=None,
                 _feat_idx=None,
                 _feat_ones=False,
                 _label_idx=None,
                 _normalize=False,
                 _binary=False,
                 _add_range=False,
                 _voxel_size=0.05):

        super().__init__()
        self.mode = _mode
        self.root_dir = _root_dir
        self.coord_idx = _coord_idx if _coord_idx is not None else [0, 1, 2]
        self.feat_idx = _feat_idx if _feat_idx is not None else [0, 1, 2]
        self.label_idx = _label_idx if _label_idx is not None else [-1]
        self.add_range = _add_range
        self.voxel_size = _voxel_size
        self.normalize = _normalize
        self.binary = _binary
        self.coords = []
        self.features = []
        self.feat_ones = _feat_ones
        self.labels = []
        self.dataset_size = 0

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)

    def __len__(self):
        if self.dataset_size > 0:
            return self.dataset_size
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        path_to_file = os.path.join(self.root_dir, file)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        self.coords = data[:, self.coord_idx].copy()
        self.features = data[:, self.feat_idx].copy()
        self.labels = data[:, self.label_idx].copy()

        if self.feat_idx[:3] == [0, 1, 2]:
            if self.normalize:
                # XYZ suposed to be 3 first features
                xyz = self.coords.copy()
                centroid = np.mean(xyz, axis=0)
                xyz -= centroid
                furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
                xyz /= furthest_distance
                self.features[:, [0, 1, 2]] = xyz

        if self.add_range:
            range = np.sqrt(np.sum(abs(self.coords) ** 2, axis=-1))
            range = range[:, None]
            self.features = np.hstack((self.features, range))

        if self.binary:
            self.labels[self.labels > 0] = 1

        if self.feat_ones:
            self.features = np.ones((self.features.shape[0], 1)) # esto son unos


        if self.mode == 'test_no_labels':
            return (self.coords.astype(np.float32) / self.voxel_size), self.features.astype(np.float32), str("No_label")
        if self.mode == 'test':
            return (self.coords.astype(np.float32) / self.voxel_size), self.features.astype(np.float32), self.labels.astype(np.int32)
        else:
            return (self.coords.astype(np.float32) / self.voxel_size), self.features.astype(np.float32), self.labels.astype(np.int32)



class RandDataset(Dataset):
  def __init__(self, n_clouds=50, n_points=3000, n_features=3):
    super(RandDataset, self).__init__()
    # do stuff here?
    self.values = np.random.rand(n_clouds, n_points, n_features)
    self.labels = np.random.rand(n_clouds, n_points)

  def __len__(self):
    return len(self.values)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.values[index], self.labels[index]


# DEPRECATED
"""
class minkDataset(Dataset): # DEPRECATED

    def __init__(self, mode_='train', root_dir = 'my_dataset_dir', features=None, labels=None, normalize=False, binary = False, add_range_=False, voxel_size_=0.05):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_
        self.coords = []
        self.voxel_size = voxel_size_
        self.mode = mode_

        if features is None:
            self.features = [0, 1, 2]
        else:
            self.features = features

        if labels is None:
            self.labels = [-1]
        else:
            self.labels = labels

        self.normalize = normalize
        self.binary = binary

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        file = self.dataset[index]
        path_to_file = os.path.join(self.root_dir, file)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = data[:, self.features].copy()
        # self.coords = features[:, [0,1,2]]
        self.coords = data[:, [0,1,2]].copy()

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = features[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0,1,2]] = xyz

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.mode == 'test_no_labels':
            return self.coords.astype(np.float32)/self.voxel_size, features.astype(np.float32)

        else:
            labels = data[:, self.labels]
            if self.binary:
                labels[labels > 0] = 1

            return self.coords.astype(np.float32) / self.voxel_size, features.astype(np.float32), labels.astype(np.int32)


class vis_minkDataset(Dataset): # DEPRECATED, USE MinkeDataset instead and append specific clouds to the dataset property

    def __init__(self, root_dir = 'my_dataset_dir',  common_clouds_dir='', extend_clouds=[], features=None, labels=None, normalize=False, binary = False, add_range_=False, voxel_size_=0.05):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_
        self.coords = []
        self.voxel_size = voxel_size_

        if features is None:
            self.features = [0, 1, 2]
        else:
            self.features = features

        if labels is None:
            self.labels = [-1]
        else:
            self.labels = labels

        self.normalize = normalize
        self.binary = binary

        self.dataset = []
        for file in os.listdir(common_clouds_dir):
            if file.endswith(".ply"):
                self.dataset.append(os.path.join(common_clouds_dir, file))
        if extend_clouds:
            for file in extend_clouds:
                self.dataset.append(os.path.join(self.root_dir, file))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        path_to_file = os.path.abspath(self.dataset[index])
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = np.copy(data[:, self.features])
        labels = np.copy(data[:, self.labels])
        self.coords = np.copy(features[:, [0,1,2]])

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = features[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0,1,2]] = xyz

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.binary:
            labels[labels > 0] = 1


        return self.coords.astype(np.float32) / self.voxel_size, features.astype(np.float32), labels.astype(np.int32)


class vis_Test_Dataset(Dataset): # DEPRECATED, USE PLYDataset instead and append specific clouds to the dataset property
    def __init__(self, root_dir = 'my_dataset_dir', common_clouds_dir='', extend_clouds=[], features=None, labels=None, normalize=False, binary = False, add_range_=False, compute_weights=False):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_

        if features is None:
            self.features = [0, 1, 2]
        else:
            self.features = features

        if labels is None:
            self.labels = [-1]
        else:
            self.labels = labels

        self.normalize = normalize
        self.binary = binary

        self.dataset = []
        for file in os.listdir(common_clouds_dir):
            if file.endswith(".ply"):
                self.dataset.append(os.path.join(common_clouds_dir, file))
        for file in extend_clouds:
            self.dataset.append(os.path.join(self.root_dir, file))

        if compute_weights:
            self.weights = []
            # COMPUTE WEIGHTS FOR EACH LABEL IN THE WHOLE DATASET
            print('-'*50)
            print("COMPUTING LABEL WEIGHTS")
            for file in tqdm(self.dataset):
                # READ THE FILE
                path_to_file = os.path.join(self.root_dir, file)
                ply = PlyData.read(path_to_file)
                data = ply["vertex"].data
                data = np.array(list(map(list, data)))

                # CONVERT TO BINARY LABELS
                labels = data[:, self.labels]
                if self.binary:
                    labels[labels > 0] = 1

                labels = np.sort(labels, axis=None)
                k_lbl, weights = np.unique(labels, return_counts=True)
                # SI SOLO EXISTE UNA CLASE EN LA NUBE (SOLO SUELO)
                if k_lbl.size < 2:
                    if k_lbl[0] == 0:
                        weights = np.array([1, 0])
                    else:
                        weights = np.array([0, 1])
                else:
                    weights = weights / len(labels)

                if len(self.weights) == 0:
                    self.weights = weights
                else:
                    self.weights = np.vstack((self.weights, weights))

            self.weights = np.mean(self.weights, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path_to_file = os.path.abspath(self.dataset[index])
        # path_to_file = os.path.join(self.root_dir, file)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = data[:, self.features]
        labels = data[:, self.labels]

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = features[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0,1,2]] = xyz

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.binary:
            labels[labels > 0] = 1

        # COMPUTE WEIGHTS FOR EACH LABEL
        # labels = np.sort(labels, axis=None)
        # _, weights = np.unique(labels, return_counts=True)
        # weights = weights/len(labels)

        return features, labels, os.path.basename(path_to_file).split('.')[0]

"""



if __name__ == '__main__':

    ROOT_DIR = os.path.abspath('/media/arvc/data/datasets/ARVCTRUSS/train/ply_xyzlabelnormal')


    NORMALIZANDO = True
    BINARY = True
    ADD_RANGE = False
    COORDENADAS = [0, 1, 2]
    FEATURES = [0, 1, 2]
    LABELS = [3]

    dataset = PLYDataset(_mode='train',
                            _root_dir=ROOT_DIR,
                            _coord_idx=COORDENADAS,
                            _feat_idx=FEATURES,
                            _label_idx=LABELS,
                            _normalize=NORMALIZANDO,
                            _binary=BINARY,
                            _add_range=ADD_RANGE,
                            _compute_weights=False)
                            


    loader_v1 = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=True)
    
    for i in range(5):

        print('-'*10)



