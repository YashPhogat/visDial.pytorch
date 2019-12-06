import numpy as np

probs = np.array([[[0.1, 0.4, 0.6], [0.2, 0.3, 0.5], [0.2, 0.8, 0.0], [0.95, 0.01, 0.04]],
                           [[0.5, 0.4, 0.1], [0.97, 0.1, 0.2], [0.3, 0.9, 0.1], [0.8, 0.15, 0.05]]])
entailment_thresh = 0.5
entail_mask = np.zeros(probs.size, dtype=np.bool)
entail_mask[:, :, :, 1] = True
thresh_mask = np.where(probs>entailment_thresh)
decrease_mask = np.where(probs<=entailment_thresh)
entail_thresh_mask = entail_mask*thresh_mask
entail_decrease_mask = decrease_mask
tags = np.zeros(shape=(probs.shape[0], probs.shape[1], probs.shape[2]))
tags[entail_thresh_mask] = 0

