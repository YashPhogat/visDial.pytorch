import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_home',type=str)
parser.add_argument('--early_stop_train', type=int,default=80000)
parser.add_argument('--early_stop_val', type=int, default=40000)

opt = parser.parse_args()
path_to_home = opt.path_to_home

early_stop_train = opt.early_stop_train
early_stop_val = opt.early_stop_val

sigmas = [0.1, 0.2, 0.5, 0.8, 1.0]
alphas = [0.05, 0.1, 0.2, 0.5]

val = 10
for sigma in sigmas:
    for alpha in alphas:
        os.system("python train_D.py --cuda --path_to_home '"+ path_to_home + "' --num_val 10 --early_stop " + str(early_stop_train) + " --pl_sigma " +str(sigma) + " --alpha_norm "+str(alpha) +" --exp_name 'rank_sig:{}_alpha:{}'".format(sigma,alpha))
        os.system("python eval_D.py --cuda --path_to_home '"+ path_to_home + "' --early_stop " + str(early_stop_val) + " --model_path '../train/save/rank_sig:{}_alpha:{}/epoch_5.pth'".format(sigma,alpha))