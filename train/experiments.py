import os

path_to_home = ''
sigmas = [0.1, 0.2, 0.5, 0.8, 1.0]
alphas = [0.05, 0.1, 0.2, 0.5]

for sigma in sigmas:
    for alpha in alphas:
        os.system("python train_D_lambda.py --cuda --path_to_home '"+ path_to_home + "' --pl_sigma " +str(sigma) + " --alpha_norm "+str(alpha) +" --exp_name 'sig:{}_alpha:{}'".format(sigma,alpha))
        os.system("python eval_D.py --cuda --path_to_home '"+ path_to_home + "' --model_path '../train/save/sig:{}_alpha:{}/epoch_5.pth'".format(sigma,alpha))