import h5py
import json
import numpy as np

file_path = '../script/data/visdial_params_demo.json'

# f = json.load(open(file_path, 'r'))
# itow = f['itow']
# img_info = f['img_train']
#
# img_list = []
# for i in img_info:
#     img_list.append(i['path'])
#
# print(img_list[5])

# f = h5py.File('data_demo2.hdf5','w')
np.random.seed(41)
data_matrix = 3*np.random.randn(10, 3)
print(data_matrix)
#
# # Write data to HDF5
# with h5py.File('file.hdf5', 'a') as data_file:
#     data_file['group_name_1'].resize(data_file['group_name_1'].shape[0] + data_matrix.shape[0],axis=0)
#     data_file['group_name_1'][-data_matrix.shape[0]:] = data_matrix

# Write data to HDF5
# with h5py.File('data_demo4.hdf5', 'w') as data_file:
#     data_file.create_dataset('a', data=data_matrix, chunks=True, maxshape=(None,3))


f = h5py.File('data_demo4.hdf5', 'r+')
x = f['a']
# # print(x[0:-1,:])
# # x[0:-1,:] = x[1:,:]
# # print(x[:,:])
# x.resize((4,3))
# print(x[:,:])
x.resize((9,3))
print(x[:,:])