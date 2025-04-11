"""
Module to run all tasks regarding first assignment.

How you manage this module is not as important. You can choose whether you do as is written in the
skeleton or if you prefer to have input-values from the command line. However, keep the names of
all methods in the lasso-module identical as we can then correct the assignment easier.
"""

import argparse
from lasso import lasso_ccd, lasso_cv, multiframe_lasso_cv, lasso_denoise
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sounddevice as sd



def plot_reconstruction(n, t, y, y_interp, ninterp, lambda_val):
    plt.figure(figsize=(8, 5))
    plt.plot(n, t, 'o', label='Original data', markersize=6)
    plt.plot(n, y, 'x', label='Reconstruction', markersize=6)
    plt.plot(ninterp, y_interp, '-', label='Interpolated reconstruction', linewidth=2)
    plt.title(f"LASSO Reconstruction (λ = {lambda_val})")
    plt.xlabel('Time index (n)')
    plt.ylabel('Amplitude')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_task5(lambda_vector, rmse_val, rmse_est, lambda_opt):
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_vector, rmse_val, 'o-', label='Validation RMSE')
    plt.plot(lambda_vector, rmse_est, 's-', label='Estimation RMSE')
    plt.axvline(lambda_opt, color='k', linestyle='--', label=f'λ_opt = {lambda_opt:.4f}')

    plt.xscale('log')
    plt.xlabel('λ (log scale)')
    plt.ylabel('RMSE')
    plt.title('Cross-Validation for LASSO')
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
def plot_task6(lambda_grid, rmse_val, rmse_est, lambda_opt):
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_grid, rmse_val, label='Validation RMSE', marker='o')
    plt.plot(lambda_grid, rmse_est, label='Estimation RMSE', marker='s')
    plt.axvline(lambda_opt, linestyle='--', color='k', label=f'λ_opt = {lambda_opt:.4f}')
    plt.xscale('log')
    plt.xlabel('λ (log scale)')
    plt.ylabel('RMSE')
    plt.title('Multi-frame LASSO Cross-Validation')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def task4():
    """
    Runs code for task 4
    """
    data = scipy.io.loadmat('A1_data.mat')
    
    X = data['X']
    t = data['t']
    
    lambda_val1 = 0.1
    lambda_val2 = 10
    lambda_val3 = (lambda_val1 + lambda_val2) / 2  # thought the average would be a suitable lambda but wasn't

    w_old = None
    
    w_hat1 = lasso_ccd(t, X, lambda_val1, w_old)
    w_hat2 = lasso_ccd(t, X, lambda_val2, w_old)
    w_hat3 = lasso_ccd(t, X, lambda_val3, w_old)
    
    n = data['n'].flatten()
    ninterp = data['ninterp'].flatten()
    Xinterp = data['Xinterp']
    
    y1 = X @ w_hat1
    y2 = X @ w_hat2
    y3 = X @ w_hat3
    
    y1_interp = Xinterp @ w_hat1
    y2_interp = Xinterp @ w_hat2
    y3_interp = Xinterp @ w_hat3


    print("Task 4")
    # print("LASSO estimate for ccd:", w_hat1, w_hat2, w_hat3)
    plot_reconstruction(n, t, y1, y1_interp, ninterp, lambda_val1)
    plot_reconstruction(n, t, y2, y2_interp, ninterp, lambda_val2)
    plot_reconstruction(n, t, y3, y3_interp, ninterp, lambda_val3)
    
    nonzero_w1 = np.sum(np.abs(w_hat1) > 0)
    nonzero_w2 = np.sum(np.abs(w_hat2) > 0)
    nonzero_w3 = np.sum(np.abs(w_hat3) > 0)
    
    print(f"λ = {lambda_val1} had {nonzero_w1} non-zero coordinates")
    print(f"λ = {lambda_val2} had {nonzero_w2} non-zero coordinates")
    print(f"λ = {lambda_val3} had {nonzero_w3} non-zero coordinates")


def task5():
    """
    Runs code for task 5
    """
    data = scipy.io.loadmat('A1_data.mat')
    X = data['X']
    t = data['t']
    n = data['n'].flatten()
    ninterp = data['ninterp'].flatten()
    Xinterp = data['Xinterp']
    
    lambda_vector = np.logspace(-3, 3, 100)
    nbr_folds = 5
    
    w_opt, lambda_opt, rmse_val, rmse_est = lasso_cv(t, X, lambda_vector, nbr_folds)

    print("Task 5")
    plot_task5(lambda_vector, rmse_val, rmse_est, lambda_opt)
    print(f"Optimal λ: {lambda_opt}")
    y_opt = X @ w_opt
    y_opt_interp = Xinterp @ w_opt
    plot_reconstruction(n, t, y_opt, y_opt_interp, ninterp, lambda_opt)

def task6():
    """
    Runs code for task 6
    """
    data = scipy.io.loadmat('A1_data.mat')
    Ttrain = data['Ttrain']
    Xaudio = data['Xaudio']    
    lambda_min = 0.001
    lambda_max = 10
    num_lambda = 5
    lambda_grid = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), num_lambda))
    
    nbr_folds = 5
    w_opt, lambda_opt, rmse_val, rmse_est = multiframe_lasso_cv(Ttrain, Xaudio, lambda_grid, nbr_folds)
    
    print("Task 6")
    plot_task6(lambda_grid, rmse_val, rmse_est, lambda_opt)
    print("Optimal λ:", lambda_opt)

def task7():
    """
    Runs code for task 7
    """
    
    data = scipy.io.loadmat("A1_data.mat")
    Ttest = data["Ttest"]
    Xaudio = data["Xaudio"]
    fs = int(data["fs"].squeeze())

    lambda_opt = 0.010000000000000004
    lambda_low = 0.0001
    lambda_medium = 0.1
    lambda_high = 0.5
    
    Ytest_opt = lasso_denoise(Ttest, Xaudio, lambda_opt)
    Ytest_low = lasso_denoise(Ttest, Xaudio, lambda_low)
    Ytest_medium = lasso_denoise(Ttest, Xaudio, lambda_medium)
    Ytest_high = lasso_denoise(Ttest, Xaudio, lambda_high)

    scipy.io.savemat("denoised_audio_opt.mat", {"Ytest": Ytest_opt, "fs": fs})
    scipy.io.savemat("denoised_audio_low.mat", {"Ytest": Ytest_low, "fs": fs})
    scipy.io.savemat("denoised_audio_medium.mat", {"Ytest": Ytest_medium, "fs": fs})
    scipy.io.savemat("denoised_audio_high.mat", {"Ytest": Ytest_high, "fs": fs})
    
    print("Task 7")
    
    
def listenToAudioFiles():
    try:
        data_opt = scipy.io.loadmat("denoised_audio_opt.mat")
        data_low = scipy.io.loadmat("denoised_audio_low.mat")
        data_medium = scipy.io.loadmat("denoised_audio_medium.mat")
        data_high = scipy.io.loadmat("denoised_audio_high.mat")

        Ytest_opt = data_opt["Ytest"].astype(np.float32).flatten()
        Ytest_low = data_low["Ytest"].astype(np.float32).flatten()
        Ytest_medium = data_medium["Ytest"].astype(np.float32).flatten()
        Ytest_high = data_high["Ytest"].astype(np.float32).flatten()
        fs = int(data_opt["fs"].squeeze())

        print("Playing optimal λ audio...")
        sd.play(Ytest_opt, fs)
        sd.wait()

        print("Playing low λ audio...")
        sd.play(Ytest_low, fs)
        sd.wait()

        print("Playing medium λ audio...")
        sd.play(Ytest_medium, fs)
        sd.wait()

        print("Playing high λ audio...")
        sd.play(Ytest_high, fs)
        sd.wait()
        
    except Exception as e:
        print("Error while loading or playing audio:", e)
    


def main():
    """
    Runs a specified task given input from the user
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--task",
        choices=["4", "5", "6", "7", "8"],
        help="Runs code for selected task.",
    )
    args = parser.parse_args()
    try:
        if args.task is None:
            task = 0
        else:
            task = int(args.task)
    except ValueError:
        print("Select a valid task number")
        return

    if task == 4:
        task4()
    elif task == 5:
        task5()
    elif task == 6:
        task6()
    elif task == 7:
        task7()
    elif task == 8:
        listenToAudioFiles()
    else:
        raise ValueError("Select a valid task number")


if __name__ == "__main__":
    main()
