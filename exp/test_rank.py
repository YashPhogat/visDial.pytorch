import numpy as np

probs = np.array([[[[0.1, 0.4, 0.6], [0.2, 0.3, 0.5], [0.2, 0.8, 0.0], [0.95, 0.01, 0.04]],
                           [[0.5, 0.4, 0.1], [0.97, 0.1, 0.2], [0.3, 0.9, 0.1], [0.8, 0.15, 0.05]]]])

entail_thres = 0.5
contra_thres = 0.9

# print(probs.shape)
entail_mask = np.zeros(probs.shape,dtype=np.bool)
entail_mask[:,:,:,1] = True
# print(entail_mask)
entail_thres_mask = probs>entail_thres
# print(entail_thres_mask)
entail_final_mask = entail_mask*entail_thres_mask
# print(entail_final_mask)
entail_final_mask_sum = np.sum(entail_final_mask,axis=-1)

contra_mask = np.zeros(probs.shape,dtype=np.bool)
contra_mask[:,:,:,0] = True
contra_thres_mask = probs>contra_thres
contra_final_mask = contra_mask*contra_thres_mask
contra_final_mask_sum = 2*np.sum(contra_final_mask,axis=-1)

tags = np.zeros(shape=(probs.shape[0],probs.shape[1],probs.shape[2]),dtype=np.int8)

tags = entail_final_mask_sum + contra_final_mask_sum
tags = 2 - tags

frequency = np.zeros(shape=(probs.shape[0],probs.shape[1],probs.shape[3]),dtype=np.int32)
frequency[:,:,0] = np.sum(tags==0,axis=-1)
frequency[:, :, 1] = np.sum(tags == 1, axis=-1)
frequency[:, :, 2] = np.sum(tags == 2, axis=-1)
print(frequency)
print(tags)



# print(tags)

