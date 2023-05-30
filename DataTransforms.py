import os
import h5py
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def H5toPly(root_dir, features, binary=True):
    data_path = os.path.abspath(root_dir)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(root_dir), os.pardir))
    file_info = 'all_files.txt'
    path_to_file = os.path.join(data_path, file_info)

    with open(path_to_file) as f:
        file_list = f.readlines()

    total = 0
    actual = 0
    for file in file_list:
        file_path = os.path.join(data_path, file)
        h5f = h5py.File(file_path.strip(), 'r')
        data = np.array(h5f.get('data'))
        labels = np.array(h5f.get('label'))

        # Add labels to the same np array
        data = np.dstack((data, labels))

        for i, cloud in enumerate(tqdm(data, desc=os.path.basename(file_path).strip())):
            # cloud = cloud.flatten()
            # cloud = list(map(tuple, cloud))
            name = 'pc_' + str(total + i)
            NptoPly(cloud, features, name, binary)
            actual = i

        total = total + actual

    print(f"{total+1} .ply files stored in {parent_dir + '/ply_dataset'}")


def TxttoPly(root_dir, features, binary=True):
    data_path = os.path.abspath(root_dir)

    for _, entry in enumerate(os.scandir(data_path)):
        filename = os.path.splitext(entry.name)[0]
        cloud = np.loadtxt(entry.path, delimiter=';')
        NptoPly(cloud, features, filename, binary)


def NptoPly(data_array, out_dir, ply_name, features, binary):

    abs_file_path = os.path.join(out_dir, ply_name + '.ply')

    cloud = list(map(tuple, data_array))
    vertex = np.array(cloud, dtype=features)
    el = PlyElement.describe(vertex, 'vertex')
    if binary:
        PlyData([el]).write(abs_file_path)
    else:
        PlyData([el], text=True).write(abs_file_path)


def PlytoNp(path_to_file):
    plydata = PlyData.read(path_to_file)
    data = plydata.elements[0].data
    # nm.memmap to np.ndarray
    data = np.array(list(map(list, data)))
    array = data[:,3]
    return array


if __name__ == '__main__':
    output_dir = '/home/arvc/PycharmProjects/HKPS/results/'
    input_array = ''

    # feat_XYZRGBXnYnZn = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('xn', 'f4'),
    #             ('yn', 'f4'), ('zn', 'f4'), ('label', 'i4')]

    # feat_XYZRGB = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    feat_XYZI = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]

    array_cloud = np.load(input_array)

    NptoPly(array_cloud, output_dir, 'test_cloud', feat_XYZI, binary=False)




