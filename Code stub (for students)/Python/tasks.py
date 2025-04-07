"""
Module to run all tasks regarding first assignment.

How you manage this module is not as important. You can choose whether you do as is written in the
skeleton or if you prefer to have input-values from the command line. However, keep the names of
all methods in the lasso-module identical as we can then correct the assignment easier.
"""

import argparse
from lasso import lasso_ccd 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = scipy.io.loadmat('A1_data.mat')

def task4():
    """
    Runs code for task 4
    """
    X = data['X']
    t = data['t']
    
    lambda_val1 = 0.1
    lambda_val2 = 10
    lambda_val3 = (lambda_val1 + lambda_val2)/2

    w_old = None
    
    w_hat1 = lasso_ccd(t, X, lambda_val1, w_old)
    w_hat2 = lasso_ccd(t, X, lambda_val2, w_old)
    w_hat3 = lasso_ccd(t, X, lambda_val3, w_old)
    
    n = data['n'].flatten()
    ninterp = data['ninterp'].flatten()
    xinterp = data['xinterp']
    
    y1 = X @ w_hat1
    y2 = X @ w_hat2
    y3 = X @ w_hat3
    
    y1_interp = xinterp @ w_hat1
    y2_interp = xinterp @ w_hat2
    y3_interp = xinterp @ w_hat3


    print("Task 4")
    print("LASSO estimate for ccd:", w_hat1, w_hat2, w_hat3)

def task5():
    """
    Runs code for task 5
    """

    print("Task 5")

def task6():
    """
    Runs code for task 6
    """

    print("Task 6")


def task7():
    """
    Runs code for task 7
    """

    print("Task 7")


def main():
    """
    Runs a specified task given input from the user
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--task",
        choices=["4", "5", "6", "7"],
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
    else:
        raise ValueError("Select a valid task number")


if __name__ == "__main__":
    main()
