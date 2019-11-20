import os

path_to_home = ''
sigmas = [0.1, 0.2, 0.5, 0.8, 1.0]
alphas = [0.05, 0.1, 0.2, 0.5]
early_stop_train = 500
early_stop_val = 500
for sigma in sigmas:
    for alpha in alphas:
        os.system("python train_D.py --cuda --path_to_home '"+ path_to_home + "' --early_stop " + str(early_stop_train) + " --pl_sigma " +str(sigma) + " --alpha_norm "+str(alpha) +" --exp_name 'rank_sig:{}_alpha:{}'".format(sigma,alpha))
        os.system("python eval_D.py --cuda --path_to_home '"+ path_to_home + "' --early_stop " + str(early_stop_val) + " --model_path '../train/save/rank_sig:{}_alpha:{}/epoch_5.pth'".format(sigma,alpha))