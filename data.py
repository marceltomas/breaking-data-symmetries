"""
Data generation pipeline.
Supports random and degenerate data partitions.

Adapted from:
 - https://github.com/danielmamay/grokking/blob/main/grokking/data.py
 - https://github.com/nmallinar/rfm-grokking/blob/main/data.py
Substantial changes made for modular/abelian group operations and consistency with custom data generation pipeline.
"""

from itertools import product
import torch
import math
from math import ceil
import itertools
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
torch.set_default_dtype(torch.float64)

MULTIPLICATIVE_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "x*y": lambda x, y, _: (x, y, x * y),
}

ADDITIVE_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x−y": lambda x, y, _: (x, y, x - y),
}

ALL_OPERATIONS = {
    **ADDITIVE_MODULO_OPERATIONS,
    **MULTIPLICATIVE_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int):
    """
    For additive operations, x and y ∈ [0, p-1].
    For multiplicative operations, x and y ∈ [1, p-1].
    
    Returns:
        inputs (torch.Tensor): Samples (x, y)
        labels (torch.Tensor): Labels x ◦ y mod p for given operation.
    """
    
    values = torch.arange(1 if operation in MULTIPLICATIVE_MODULO_OPERATIONS else 0, p)
    x, y = torch.cartesian_prod(values, values).T

    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    results = z % p

    inputs = torch.stack([x, y], dim=1)
    labels = results
    return inputs, labels

def abelian_data(group_sizes): 
    """
    Given an abelian group described as a direct product of cyclic groups,
    A ≅ C_n × C_m × ..., generate all (a, b) → a + b pairs.

    Each element a ∈ A is represented as a tuple of integers modulo the corresponding group size.
    The group operation is component-wise addition modulo each cyclic group's order.

    Args:
        group_sizes (list[int]): Sizes of the cyclic groups in the direct product.

    Returns:
        inputs (torch.Tensor): Tensor of shape (n^2, 2, d) where each row is a pair (a, b), 
                               with a, b ∈ A, n is the order of A, and d = len(group_sizes).
        labels (torch.Tensor): Tensor of shape (n^2, d) with a + b (mod group_sizes) for each input pair.
    """
    
    elements = list(product(*[range(p) for p in group_sizes]))
    X = []
    y = []

    for a in elements:
        for b in elements:
            z = tuple((ai + bi) % pi for ai, bi, pi in zip(a, b, group_sizes))
            X.append([a, b])
            y.append(z)

    X = torch.tensor(X)  # shape: (n^2, 2, d)
    y = torch.tensor(y)  # shape: (n^2, d)
    return X, y

def encode_abelian(X, y, group_sizes):
    """
    Encode abelian group elements represented as tuples into integers (for one-hot encoding).
    Each tuple is flattened into a single integer using mixed-radix encoding.

    Args:
        X (torch.Tensor): Tensor (or list) of shape (n, 2, d), where d = len(group_sizes)
        y (torch.Tensor): Tensor (or list) of shape (n, d)
        group_sizes (list[int]): Sizes of the cyclic groups in the direct product

    Returns:
        X_encoded (torch.Tensor): Tensor of shape (n, 2) with integer-encoded pairs
        y_encoded (torch.Tensor): Tensor of shape (n,) with integer-encoded results
    """
    def encode_single(x):
        result = 0
        multiplier = 1
        for xi, pi in zip(reversed(x), reversed(group_sizes)):
            result += xi * multiplier
            multiplier *= pi
        return result

    X_encoded = torch.tensor([
        [encode_single(a), encode_single(b)]
        for a, b in X
    ])
    if y is not None:
        y_encoded = torch.tensor([
            encode_single(yi) for yi in y
        ])
    else:
        y_encoded = None
    return X_encoded, y_encoded


def decode_abelian(X, y, group_sizes):
    """
    Decode integer-encoded abelian group elements back to tuple form.

    Args:
        X (torch.Tensor): Tensor of shape (n, 2) with integer-encoded inputs
        y (torch.Tensor): Tensor of shape (n,) with integer-encoded outputs
        group_sizes (list[int]): Sizes of the cyclic groups in the direct product

    Returns:
        X_decoded (torch.Tensor): Tensor of shape (n, 2, d) with tuple form
        y_decoded (torch.Tensor): Tensor of shape (n, d) with tuple form
    """
    def decode_single(code):
        coords = []
        for pi in reversed(group_sizes):
            coords.append(code % pi)
            code //= pi
        return tuple(reversed(coords))

    X_decoded = torch.tensor([
        [decode_single(a.item()), decode_single(b.item())]
        for a, b in X
    ])
    if y is not None:
        y_decoded = torch.tensor([
            decode_single(yi.item()) for yi in y
        ])
    else:
        y_decoded = None
    
    return X_decoded, y_decoded

def decode_one_hot(X):
    """
    Decodes a one-hot encoded tensor of shape (n, 2*p) into integer pairs (a, b) of shape (n,2).
    """

    device = X.device
    p = X.shape[1] // 2

    a_indices = X[:, :p].argmax(dim=1)
    b_indices = X[:, p:].argmax(dim=1)

    return torch.stack([a_indices, b_indices], dim=1).to(device)

def make_data_splits(inputs, labels, training_fraction):
    """
    Given a training_fraction, partitions inputs and labels into train and test sets.
    """
    
    train_size = int(training_fraction * inputs.shape[0])
    val_size = inputs.shape[0] - train_size

    perm = torch.randperm(inputs.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]

def move_points_between_sets(X_tr, y_tr, X_te, y_te, n_test_to_train, n_train_to_test):
    """
    Moves random samples between test and train sets, preserving input-label pairing.
    """
    # Move samples from test to train
    if n_test_to_train > 0:
        idx = torch.randperm(len(X_te))[:n_test_to_train]
        X_tr = torch.cat([X_tr, X_te[idx]], dim=0)
        y_tr = torch.cat([y_tr, y_te[idx]], dim=0)
        mask = torch.ones(len(X_te), dtype=torch.bool)
        mask[idx] = False
        X_te = X_te[mask]
        y_te = y_te[mask]

    # Move samples from train to test
    if n_train_to_test > 0:
        idx = torch.randperm(len(X_tr))[:n_train_to_test]
        X_te = torch.cat([X_te, X_tr[idx]], dim=0)
        y_te = torch.cat([y_te, y_tr[idx]], dim=0)
        mask = torch.ones(len(X_tr), dtype=torch.bool)
        mask[idx] = False
        X_tr = X_tr[mask]
        y_tr = y_tr[mask]

    return X_tr, y_tr, X_te, y_te

def partition_fixed_points(X, y, operation: str, p, reflections):
    """
    Partitions dataset into fixed points and the rest, under one or more reflections sr^k.

    Args:
        X (torch.Tensor):
            - Modular case: shape (n, 2), each row is (a, b) (additive: a and b ∈ [0, p-1], multiplicative: a and b ∈ [1, p-1])
            - Abelian case: shape (n, 2, d), each row is (a_tuple, b_tuple)
        y (torch.Tensor): 
            - Modular: shape (n,)
            - Abelian: shape (n, d)
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)
        p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        reflections (int or list[int]): One or more reflection indices k. Used to determine fixed points

    Returns:
        X_tr, y_tr: training data (non-fixed points)
        X_te, y_te: test data (fixed points under at least one reflection)
    """
    if isinstance(reflections, (int, float)):
        reflections = [int(reflections)]

    # Abelian case: p is list[int], X has shape (n, 2, d)
    if isinstance(p, list) or operation == "abelian":
        p_tensor = torch.tensor(p, dtype=X.dtype, device=X.device)
        d = X.shape[-1]

        is_fixed_point = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
        
        if isinstance(reflections[0], (int, float)): reflections = [reflections] # Normalize the case for a single reflection [k1,k2,...]
        for k in reflections:
            k_tensor = torch.tensor(k, dtype=X.dtype, device=X.device)
            a = X[:, 0]  # (n, d)
            b = X[:, 1]  # (n, d)
            cond = (b == (a + k_tensor) % p_tensor).all(dim=1)
            is_fixed_point |= cond

    # Modular case: p is int, X has shape (n, 2)
    else:
        a = X[:, 0]
        b = X[:, 1]
        is_fixed_point = torch.zeros(len(X), dtype=torch.bool, device=X.device)

        for k in reflections:
            if operation == "x+y":
                cond = (b == (a + k) % p)

            elif operation == "x-y":
                cond = (b == (-a - k) % p)

            elif operation == "x*y":
                cond = (b == (a * k) % p)

            elif operation == "x/y":
                ak = (a * k) % p
                ak_inv = torch.full_like(ak, -1)
                nonzero = ak != 0
                ak_inv[nonzero] = torch.tensor(
                    [pow(x.item(), -1, p) for x in ak[nonzero]],
                    dtype=ak.dtype,
                    device=X.device
                )
                cond = (b == ak_inv)

            else:
                raise ValueError(f"Unsupported operation: {operation}")

            is_fixed_point |= cond

    # Partition
    X_te = X[is_fixed_point]
    y_te = y[is_fixed_point]
    X_tr = X[~is_fixed_point]
    y_tr = y[~is_fixed_point]

    return X_tr, y_tr, X_te, y_te

def is_multiple_reflections_case(reflections, operation):
    if isinstance(reflections, (int, float)):
        reflections = [int(reflections)]
        
    if operation == "abelian":
        return hasattr(reflections[0], '__len__') and len(reflections) > 1
    return len(reflections) > 1

def move_reflected_pairs_to_test(X_tr, y_tr, X_te, y_te, operation, p, reflection, n_pairs_to_test):
    """
    Moves reflected pairs under a single reflection sr^k from the training set to the test set.

    Args:
        X_tr, X_te: Train and test inputs
            - Modular case: shape (n, 2), each row is (a, b) (additive: a and b ∈ [0, p-1], multiplicative: a and b ∈ [1, p-1])
            - Abelian case: shape (n, 2, d), each row is (a_tuple, b_tuple)
        y_tr, y_te: Train and test labels
            - Modular: shape (n,)
            - Abelian: shape (n, d)
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)
         p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        reflections (int or list[int]): Single reflection index k or [k1, ..., kd] for abelian
        n_pairs_to_test (int): Number of such pairs to move from train to test

    Returns:
        Updated X_tr, y_tr, X_te, y_te
    """
    if isinstance(reflection, (int, float)):
        reflection = [int(reflection)]
        
    if is_multiple_reflections_case(reflection, operation):
        raise NotImplementedError(
            "Moving reflected pairs is only supported for a single reflection.\n"
            "For multiple reflections, you'd need to compute orbits under the dihedral subgroup "
            "generated by the reflections, which is not implemented."
        )

    device = X_tr.device
    is_abelian = operation == "abelian"

    for _ in range(n_pairs_to_test):
        if len(X_tr) == 0:
            break

        idx = torch.randint(len(X_tr), (1,)).item()

        if is_abelian:
            a, b = X_tr[idx]
            p_tensor = torch.tensor(p, dtype=X_tr.dtype, device=device)
            k_tensor = torch.tensor(reflection, dtype=X_tr.dtype, device=device)
            
            a_ref = (b - k_tensor) % p_tensor
            b_ref = (a + k_tensor) % p_tensor
            pair = torch.stack([a_ref, b_ref]).to(device)
            
            matches = (X_tr == pair).all(dim=2).all(dim=1).nonzero(as_tuple=True)[0]
            
        else:
            a, b = X_tr[idx]

            k = reflection[0]  # scalar
            if operation == "x+y":
                a_ref = (b.item() - k) % p
                b_ref = (a.item() + k) % p

            elif operation == "x-y":
                a_ref = (-b.item() - k) % p
                b_ref = (-a.item() - k) % p

            elif operation == "x*y":
                a_ref = (b.item() * pow(k, -1, p)) % p
                b_ref = (a.item() * k) % p

            elif operation == "x/y":
                a_ref = pow(b.item() * k, -1, p)
                b_ref = pow(a.item() * k, -1, p)

            else:
                raise ValueError(f"Unsupported operation: {operation}")

            pair = torch.tensor([a_ref, b_ref], dtype=torch.long, device=device)
            matches = (X_tr == pair).all(dim=1).nonzero(as_tuple=True)[0]

        if len(matches) == 0:
            continue  

        jdx = matches[0].item()
        
        X_te = torch.cat([X_te, X_tr[[idx]], X_tr[[jdx]]], dim=0)
        y_te = torch.cat([y_te, y_tr[[idx]], y_tr[[jdx]]], dim=0)

        mask = torch.ones(len(X_tr), dtype=torch.bool, device=device)
        mask[[idx, jdx]] = False
        X_tr = X_tr[mask]
        y_tr = y_tr[mask]
        

    return X_tr, y_tr, X_te, y_te


def degenerate_data_generator(operation: str, p, reflections, n_test_to_train: int = 0, n_train_to_test: int = 0, n_pairs_to_test: int = 0):
    """
    Generate a degenerate training/test split for modular or abelian group operations.

    Args:
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)
        p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        reflections (int or list[int]): One or more reflection indices k. Used to determine fixed points
        n_test_to_train (int): Number of random points to move from test to train
        n_train_to_test (int): Number of random points to move from train to test
        n_pairs_to_test (int): Number of reflected pairs to move from test to train (only supported if len(reflections) == 1)

    Returns:
        X_tr (torch.Tensor): One-hot encoded training inputs of shape (n_train, 2 * p)
        y_tr (torch.Tensor): One-hot encoded training labels of shape (n_train, p)
        X_te  (torch.Tensor): One-hot encoded test inputs of shape (n_test, 2 * p)
        y_te  (torch.Tensor): One-hot encoded test labels of shape (n_test, p)
    """
    is_abelian = operation == "abelian"
    is_multiplicative = operation in MULTIPLICATIVE_MODULO_OPERATIONS
    if isinstance(reflections, (int, float)):
        reflections = [int(reflections)]
    
    if is_abelian:
        X, y = abelian_data(p)
        total_size = math.prod(p)
    else:
        X, y = operation_mod_p_data(operation, p)
        total_size = (p - 1) if is_multiplicative else p
       
    X_tr, y_tr, X_te, y_te = partition_fixed_points(X, y, operation, p, reflections)
    X_tr, y_tr, X_te, y_te = move_points_between_sets(X_tr, y_tr, X_te, y_te, n_test_to_train, n_train_to_test)

    if n_pairs_to_test > 0 and not is_multiple_reflections_case(reflections, operation):
        X_tr, y_tr, X_te, y_te = move_reflected_pairs_to_test(X_tr, y_tr, X_te, y_te, operation, p, reflections, n_pairs_to_test)
        
    elif n_pairs_to_test > 0 and is_multiple_reflections_case(reflections, operation):
        raise NotImplementedError(
            "Moving reflected pairs is only supported for a single reflection.\n"
            "For multiple reflections, you'd need to compute orbits under the dihedral subgroup "
            "generated by the reflections, which is not implemented."
        )
        
    if is_abelian:
        X_tr, y_tr = encode_abelian(X_tr, y_tr, p)
        X_te, y_te = encode_abelian(X_te, y_te, p)

    elif is_multiplicative:
        X_tr = X_tr - 1; y_tr = y_tr - 1
        X_te = X_te - 1; y_te = y_te - 1

    # One-hot encode
    X_tr = F.one_hot(X_tr, total_size).view(-1, 2 * total_size).double()
    y_tr = F.one_hot(y_tr, total_size).double()
    X_te = F.one_hot(X_te, total_size).view(-1, 2 * total_size).double()
    y_te = F.one_hot(y_te, total_size).double()

    return X_tr, y_tr, X_te, y_te

def random_partition_generator(operation: str, p, training_fraction: float):
    """
    Generate one-hot encoded training and test data for a given modular operation or abelian group.
    Supports operations in ALL_OPERATIONS or the special case "abelian" for abelian group addition.

    Args:
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)
        p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        training_fraction (float): Fraction of data to include in the training split

    Returns:
        X_tr (torch.Tensor): One-hot encoded training inputs of shape (n_train, 2 * p)
        y_tr (torch.Tensor): One-hot encoded training labels of shape (n_train, p)
        X_te  (torch.Tensor): One-hot encoded test inputs of shape (n_test, 2 * p)
        y_te  (torch.Tensor): One-hot encoded test labels of shape (n_test, p)
    """
    is_multiplicative = operation in MULTIPLICATIVE_MODULO_OPERATIONS

    if operation == "abelian":
        X, y = abelian_data(p)
        X, y = encode_abelian(X, y, p)
        total_size = math.prod(p)
        
    else:
        X, y = operation_mod_p_data(operation, p) 
        if operation in MULTIPLICATIVE_MODULO_OPERATIONS:
            # Shift values down by 1 to make them 0-based
            p -= 1
            X = X - 1
            y = y - 1
            
        total_size = p

    X_tr, y_tr, X_te, y_te = make_data_splits(X, y, training_fraction)

    # One-hot encode
    X_tr = F.one_hot(X_tr, total_size).view(-1, 2 * total_size).double()
    y_tr = F.one_hot(y_tr, total_size).double()
    X_te = F.one_hot(X_te, total_size).view(-1, 2 * total_size).double()
    y_te = F.one_hot(y_te, total_size).double()

    return X_tr, y_tr, X_te, y_te
