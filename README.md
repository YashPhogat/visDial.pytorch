# Visual Dialog using GAN
Visual Dialog model trained by adversarial approach with multiple discriminator for feedback

### Introduction
This code contains our own model modifications including Logician and Lambda Ranking. It is built on the 
 base code for the paper ["Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model"](https://arxiv.org/abs/1706.01554)
Please note that the Logician training code is in a different repository. After training it,
we have classified all 100 answer options against the ground truth answer for the first 8000 images
of the training set and use those probabilities directly here to speedup the training process.
### Dependencies

1. PyTorch. Install [PyTorch](http://pytorch.org/) with proper commands. Make sure you also install *torchvision*.
2. h5py. ```pip install h5py```
3. progressbar. ```pip install progressbar```
### Evaluation

* The preprocessed feature can be found [here](https://drive.google.com/open?id=1HFEbt0cld0QNYASLBJ_pC3nXI9gv-WpE)
* The pre-trained model can be found [here](https://drive.google.com/open?id=19IyQzwRrEieewlxu-MjyxE388hCsjJSY)