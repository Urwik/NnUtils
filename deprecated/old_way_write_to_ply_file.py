import h5py
import os

DIR = '/home/arvc/PycharmProjects/S3DIS_dataset/indoor3d_sem_seg_hdf5_data'

with open(DIR + '/all_files.txt') as f:
    file_list = f.readlines()

with open(DIR + '/room_filelist.txt') as f:
    room_list = f.readlines()

# files = []
# for filename in os.scandir(DIR): # type: os.DirEntry
#     name, ext = os.path.splitext(filename.path)
#     if ext == ".h5":
#         files.append(filename.path)

# h5f = h5py.File(files[0], 'r')
# TOTAL_PC = h5f['data'][0].size * len(files)
# TOTAL_PC = len(str(TOTAL_PC))
i = 0
for file in file_list:
    filepath = os.path.abspath('/home/arvc/PycharmProjects/S3DIS_dataset/' + file)
    with h5py.File(filepath.strip(), 'r') as h5f:
        data = h5f['data']
        labels = h5f['label']
        for cnt, cloud in enumerate(data):
            with open(f'/home/arvc/PycharmProjects/S3DIS_dataset/files/pcds_{str(i)}.ply','w') as ply_f:
                ply_f.write('ply\n')
                ply_f.write('format ascii 1.0\n')
                ply_f.write(f'comment XYZ RGB XYZNormalized Area\n')
                ply_f.write(f'element vertex {len(data[1])}\n')
                ply_f.write('property float x\n')
                ply_f.write('property float y\n')
                ply_f.write('property float z\n')
                ply_f.write('property uchar R\n')
                ply_f.write('property uchar G\n')
                ply_f.write('property uchar B\n')
                ply_f.write('property float xn\n')
                ply_f.write('property float yn\n')
                ply_f.write('property float zn\n')
                ply_f.write('end_header\n')
                for point in cloud:
                    ply_f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\
 {point[3]:.6f} {point[4]:.6f} {point[5]:.6f}\
 {point[6]:.6f} {point[7]:.6f} {point[8]:.6f}')
            i += 1
