import torch
import numpy as np

probs = torch.FloatTensor([[[0.1, 0.4, 0.6], [0.2, 0.3, 0.5], [0.2, 0.8, 0.0], [0.95, 0.01, 0.04]],
                           [[0.3, 0.4, 0.3], [0.97, 0.1, 0.2], [0.3, 0.6, 0.1], [0.23, 0.69, 0.08]]])

max_ind = probs.argmax(dim=2)

one_hot = torch.nn.functional.one_hot(max_ind,3).double()

print(max_ind.detach().numpy())
print(one_hot)

alphaC = 2.0
alphaE = 0.1
alphaN = 1.0

w = one_hot[:, :, 0]*alphaC + one_hot[:, :, 1]*alphaE + one_hot[:, :, 2]*alphaN
print(w)

sw = torch.sum(w, dim = 1)
print(sw)

dist_summary = torch.sum(torch.sum(one_hot, dim = 1), dim = 0)
smooth_dist_summary = torch.sum(torch.sum(probs, dim=1), dim=0)
print(dist_summary)
print(smooth_dist_summary)