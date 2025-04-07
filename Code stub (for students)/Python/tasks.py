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

def task4():
    """
    Runs code for task 4
    """
    data = scipy.io.loadmat('A1_data.mat')
    X = data['X']
    t = data['t']
    lambda_val = 0.1
    w_old = None
    w_hat = lasso_ccd(t, X, lambda_val, w_old)

    print("Task 4")
    print("LASSO estimate for ccd:", w_hat)

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
