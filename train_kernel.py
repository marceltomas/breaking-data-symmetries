"""
Kernel training pipeline using RFM (Recursive Feature Machine).
Supports Gaussian and Quadratic kernels with learned feature matrices M.

Based on:
https://github.com/nmallinar/rfm-grokking/blob/main/train_kernel.py
Adapted for modular and abelian group experiments.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random
import easydict

from models import gaussian_kernel, quadratic_kernel
from data import random_partition_generator, degenerate_data_generator

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def get_test_kernel(X_tr, X_te, M, bandwidth, ntk_depth, kernel_type):
    """
    Computes the test kernel matrix K(X_tr, X_te) using the learned kernel.
    """
    device = X_tr.device
    X_te = X_te.to(device)
    M = M.to(device)
    
    K_test = None
    if kernel_type == 'gaussian':
        K_test = gaussian_kernel.gaussian_M(X_tr, X_te, bandwidth, M)
    elif kernel_type == 'quadratic':
        K_test = quadratic_kernel.quadratic_M(X_tr, X_te, M)
        
    return K_test

def solve(X_tr, y_tr_onehot, M, bandwidth, ntk_depth, kernel_type,
          ridge=1e-3):
    """
    Solves the kernel ridge regression problem: (K + λI)^-1 Y.
    """
    K_train = None
    dist = None # Placeholder for Laplace kernel
    sol = None
    
    if kernel_type == 'gaussian':
        K_train = gaussian_kernel.gaussian_M(X_tr, X_tr, bandwidth, M)
    elif kernel_type == 'quadratic':
        K_train = quadratic_kernel.quadratic_M(X_tr, X_tr, M)
        
    device = K_train.device
    y_tr_onehot = y_tr_onehot.to(device)

    n = K_train.shape[0]
    ridge_matrix = ridge * torch.eye(n, device=device, dtype=K_train.dtype)
    K_ridge = K_train + ridge_matrix

    sol = torch.linalg.solve(K_ridge, y_tr_onehot)

    return sol, K_train, dist

def update(samples, centers, bandwidth, M, weights, K, dist, \
           kernel_type, ntk_depth, centers_bsize=-1, centering=False,
           agop_power=0.5):
    """
    Performs the AGOP update step for the kernel matrix M.
    """
    if kernel_type == 'gaussian':
        M, per_class_agops = gaussian_kernel.gaussian_M_update(samples, centers, bandwidth, M, weights, K=K, \
                              centers_bsize=centers_bsize, centering=centering, agop_power=agop_power,
                              return_per_class_agop=False)
    elif kernel_type == 'quadratic':
        M, per_class_agops = quadratic_kernel.quad_M_update(samples, centers, weights.T, M, centering=centering,
                                                            return_per_class_agop=False)
    return M, per_class_agops

def eval(sol, K, y_onehot):
    """
    Evaluates predictions using the learned kernel solution.
    """
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0 # placeholder
    if y_onehot.shape[1] > 1:
        count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
        acc = count / y_onehot.shape[0]
    else:
        acc = 0.0

    return acc, loss, corr

def rfm(args):
    """
    Run RFM training on a fixed dataset using a kernel regression approach.

    Parameters (from args):
        X_tr (torch.Tensor): Training inputs (one-hot encoded), shape (n_train, d)
        y_tr (torch.Tensor): Training targets (one-hot encoded), shape (n_train, d/2)
        X_te (torch.Tensor, optional): Test inputs, shape (n_test, d). Defaults to Xtr
        y_te (torch.Tensor, optional): Test targets, shape (n_test, d/2). Defaults to ytr
        M (torch.Tensor, optional): Initial feature matrix for kernel. If None, initializes as identity
        iters (int): Number of RFM update steps
        ridge (float): Regularization strength for kernel regression
        bandwidth (float): Bandwidth parameter for Gaussian kernel (if used)
        kernel_type (str): Type of kernel ('gaussian' or 'quadratic')
        print_progress (bool): Whether to print loss/accuracy during training

    Notes:
        - ntk_depth is included in args but currently unused.
        - agop_power is built-in to update function but not varied in practice.

    Returns:
        dict with the following keys:
            - 'train_accs': List of training accuracies per iteration
            - 'train_losses': List of training losses per iteration
            - 'test_accs': List of test accuracies per iteration
            - 'test_losses': List of test losses per iteration
            - 'M_list': List of M matrices after each iteration
            - 'predictions': Final predicted logits on test set
            - 'solution': Kernel regression solution
    """
    tr_accs = []
    tr_losses = []
    te_accs = []
    te_losses = []
    Mlist = []

    device = args.device

    # Load train and test sets
    X_tr = args.X_tr.to(device)
    y_tr = args.y_tr.to(device)
    X_te = args.X_te.to(device) if args.X_te is not None else X_tr
    y_te = args.y_te.to(device) if args.y_te is not None else y_tr

    # Initialize feature matrix M
    M = args.M.to(device) if args.M is not None else torch.eye(X_tr.shape[1], dtype=torch.float64, device=device)

    for rfm_iter in range(args.iters):
        # Step 1: Kernel regression solution
        sol, K_train, dist = solve(X_tr, y_tr, M, args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge)

        # Training metrics
        acc, loss, _ = eval(sol, K_train, y_tr)
        tr_accs.append(acc)
        tr_losses.append(loss)

        if args.print_progress:
            print(f'Round {rfm_iter} Train MSE:\t{loss}')
            print(f'Round {rfm_iter} Train Acc:\t{acc}')

        # Test metrics
        K_test = get_test_kernel(X_tr, X_te, M, args.bandwidth, args.ntk_depth, args.kernel_type)
        acc, loss, _ = eval(sol, K_test, y_te)
        te_accs.append(acc)
        te_losses.append(loss)

        if args.print_progress:
            print(f'Round {rfm_iter} Test MSE:\t{loss}')
            print(f'Round {rfm_iter} Test Acc:\t{acc}')
            print()

        # Step 2: Feature matrix update
        M, _ = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist,
                      args.kernel_type, args.ntk_depth,
                      centers_bsize=-1, centering=True)

        Mlist.append(M)

    # Final predictions on test set
    preds = K_test.T @ sol

    return easydict.EasyDict({
        "train_accs": tr_accs,
        "train_losses": tr_losses,
        "test_accs": te_accs,
        "test_losses": te_losses,
        "M_list": Mlist,
        "predictions": preds,
        "solution": sol
    })

def get_reflection_M(operation: str, p, k, device):
    # Returns feature matrix corresponding to reflection sr^k
    X_tr, y_tr, X_te, y_te = degenerate_data_generator(operation, p, k, 
                                                           n_test_to_train = 0, n_train_to_test = 0, n_pairs_to_test = 0)

    args = easydict.EasyDict({
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "y_te": y_te,
        "M": None,
        "iters": 50,
        "ridge": 0.0,
        "bandwidth": 2.5,
        "ntk_depth": 2,
        "kernel_type": "gaussian",
        "print_progress": False,
        "device": device
    })
        
    results = rfm(args)
    return results["M_list"][-1]

def train(data_args, rfm_args):
    """
    Runs full training pipeline: data generation + RFM training.

    Parameters:
        data_args (dict or EasyDict): Configuration for data generation.
            - operation (str): Operation ("x+y", "x*y", "abelian", etc.)
            - prime (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
            - partition_type (str): 'random' or 'degenerate'
            - training_fraction (float): Fraction of data to include in the training split (random)
            - reflections (int or list[int]): One or more reflection indices k. Used to determine fixed points (degenerate)
            - n_test_to_train (int): Number of random points to move from test to train (degenerate)
            - n_train_to_test (int): Number of random points to move from train to test (degenerate)
            - n_pairs_to_test (int): Number of reflected pairs to move from test to train (degenerate)
            - M_reflection (int or None): If set, uses reflection-k trained M as RFM init
        rfm_args (dict or EasyDict): Configuration for RFM training.

    Returns:
        dict: RFM training outputs, same structure as rfm().
    """
    # 1. Generate dataset
    if data_args.partition_type == "random":
        X_tr, y_tr, X_te, y_te = random_partition_generator(data_args.operation, data_args.prime, data_args.training_fraction)
        
    elif data_args.partition_type == "degenerate":
        X_tr, y_tr, X_te, y_te = degenerate_data_generator(data_args.operation, data_args.prime, data_args.reflections, 
                                                           data_args.n_test_to_train, data_args.n_train_to_test, data_args.n_pairs_to_test)
        
    else:
        raise ValueError(f"Unsupported partition_type: {data_args.partition_type}")

    # 2. Update RFM args with generated data
    rfm_args.X_tr = X_tr
    rfm_args.y_tr = y_tr
    rfm_args.X_te = X_te
    rfm_args.y_te = y_te

    # 3. Set initial M matrix
    if data_args.M_reflection is not None:
        rfm_args.M = get_reflection_M(data_args.operation, data_args.prime, data_args.M_reflection, rfm_args.device)
    else:
        rfm_args.M = None
        
    # 4. Run RFM training
    if rfm_args.print_progress:
        print(f"Training RFM on {len(X_tr)} train and {len(X_te)} test examples...")
    results = rfm(rfm_args)
    return results, rfm_args