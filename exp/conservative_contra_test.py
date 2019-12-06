import torch
import numpy as np

probs = torch.FloatTensor([[[0.1, 0.4, 0.6], [0.2, 0.3, 0.5], [0.2, 0.8, 0.0], [0.95, 0.01, 0.04]],
                           [[0.5, 0.4, 0.1], [0.97, 0.1, 0.2], [0.3, 0.9, 0.1], [0.8, 0.15, 0.05]]])

thresh = 0.9
thresh_mask = torch.gt(probs, 0.9)
contra_mask = torch.BoolTensor(probs.size())
contra_mask[:, :, :] = False
contra_mask[:, :, 0] = True
contra_selected = contra_mask*thresh_mask
decrease_contra_mask = contra_mask*(thresh_mask.logical_not())
print(contra_mask)
probs[decrease_contra_mask] = 0.
new_one_hot = torch.nn.functional.one_hot(probs.argmax(dim=2), 3).byte()

print(new_one_hot)