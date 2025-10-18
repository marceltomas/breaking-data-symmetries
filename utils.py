"""
Utility functions for group actions, plots and experiments
"""

import torch
import numpy as np
import torch.nn.functional as F
import math
import itertools
from itertools import product
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

from data import encode_abelian, decode_abelian, decode_one_hot, MULTIPLICATIVE_MODULO_OPERATIONS, ADDITIVE_MODULO_OPERATIONS, ALL_OPERATIONS

def print_one_hot_pairs_abelian(X, group_sizes):
    """
    Decode and print abelian group element pairs from a one-hot encoded tensor.

    Args:
        X (torch.Tensor): Tensor of shape (n_samples, 2 * n_codes),
            where n_codes = product of group_sizes.
        group_sizes (list[int]): Sizes of the cyclic groups [p1, p2, ..., pd].
    """
    device = X.device
    n_codes = X.shape[1] // 2
    a_indices = X[:, :n_codes].argmax(dim=1)
    b_indices = X[:, n_codes:].argmax(dim=1)

    # Convert indices back into abelian group tuples
    a_tensor = a_indices.unsqueeze(1).to(device)
    b_tensor = b_indices.unsqueeze(1).to(device)

    decoded, _ = decode_abelian(torch.cat([a_tensor, b_tensor], dim=1), torch.zeros_like(a_tensor), group_sizes)

    for i, (a_tuple, b_tuple) in enumerate(decoded):
        a_str = ", ".join(str(x.item()) for x in a_tuple)
        b_str = ", ".join(str(x.item()) for x in b_tuple)
        print(f"Sample {i + 1}: (({a_str}), ({b_str}))")


def print_one_hot_pairs(X, operation: str):
    """
    Decode and print (a, b) pairs from a one-hot encoded tensor.
    """
    pairs = decode_one_hot(X)
    pairs_list = pairs.cpu().tolist()

    for i, (a, b) in enumerate(pairs_list):
        if operation in ("x*y", "x/y"):
            print(f"Sample {i + 1}: ({a + 1}, {b + 1})")  # Shift back to [1, p]
        else:
            print(f"Sample {i + 1}: ({a}, {b})")
            
def split_predictions(preds, y_true, X):
    """
    Splits a dataset into correctly and incorrectly predicted samples.

    Args:
        preds (torch.Tensor): Model predictions (logits or probabilities)
        y_true (torch.Tensor): True one-hot labels.
        X (torch.Tensor): Input samples corresponding to preds/y_true.

    Returns:
        X_correct (torch.Tensor): Subset of X with correct predictions.
        X_incorrect (torch.Tensor): Subset of X with incorrect predictions.
    """
    pred_labels = preds.argmax(dim=-1)
    true_labels = y_true.argmax(dim=-1)
    correct_mask = pred_labels == true_labels

    X_correct = X[correct_mask]
    X_incorrect = X[~correct_mask]

    return X_correct, X_incorrect

def r(X, k, p, operation: str):
    """
    Rotation transformation r^k applied to batch X of shape (n, 2, d) where each row is a sample (a, b).
    """
    is_multiplicative = operation in MULTIPLICATIVE_MODULO_OPERATIONS
    if operation == "abelian" and X.dim() <= 2:   # Single sample of shape (2,d) or (2,)    
        X = X.unsqueeze(0)
    elif operation != "abelian" and X.dim() == 1: # Single sample of shape (2,)
        X = X.unsqueeze(0)    
        
    a, b = X[:, 0], X[:, 1]

    if operation == "abelian":
        p_tensor = torch.tensor(p, device=X.device)
        k_tensor = torch.tensor(k, device=X.device)
        a_ref = (a + k_tensor) % p_tensor
        b_ref = (b - k_tensor) % p_tensor
        return torch.stack([a_ref, b_ref], dim=1)

    if is_multiplicative:         # Shift back to [1, p-1]
        a, b = a + 1, b + 1

    if operation == "x+y":
        a_ref = (a + k) % p
        b_ref = (b - k) % p
    elif operation == "x-y":
        a_ref = (a + k) % p
        b_ref = (b + k) % p
    elif operation == "x*y":
        a_ref = (a * k) % p
        b_ref = (b * pow(k, -1, p)) % p
    elif operation == "x/y":
        a_ref = (a * k) % p
        b_ref = (b * k) % p
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    if is_multiplicative:
        a_ref -= 1
        b_ref -= 1

    return torch.stack([a_ref, b_ref], dim=1)


def s(X, k, p, operation: str):
    """
    Reflection transformation sr^k applied to batch X of shape (n, 2, d), where each row is a sample (a, b).
    """
    is_multiplicative = operation in MULTIPLICATIVE_MODULO_OPERATIONS
    if operation == "abelian" and X.dim() <= 2:   # Single sample of shape (2,d) or (2,)    
        X = X.unsqueeze(0)
    elif operation != "abelian" and X.dim() == 1: # Single sample of shape (2,)
        X = X.unsqueeze(0)    
        
    a, b = X[:, 0], X[:, 1]

    if operation == "abelian":
        p_tensor = torch.tensor(p, device=X.device)
        k_tensor = torch.tensor(k, device=X.device)
        a_ref = (b - k_tensor) % p_tensor
        b_ref = (a + k_tensor) % p_tensor
        return torch.stack([a_ref, b_ref], dim=1)

    if is_multiplicative:    # Shift back to [1, p-1]
        a, b = a + 1, b + 1

    if operation == "x+y":
        a_ref = (b - k) % p
        b_ref = (a + k) % p
    elif operation == "x-y":
        a_ref = (-b - k) % p
        b_ref = (-a - k) % p
    elif operation == "x*y":
        a_ref = (b * pow(k, -1, p)) % p
        b_ref = (a * k) % p
    elif operation == "x/y":
        a_ref = torch.tensor([pow(int(bi * k), -1, p) for bi in b], dtype=a.dtype, device=a.device)
        b_ref = torch.tensor([pow(int(ai * k), -1, p) for ai in a], dtype=b.dtype, device=b.device)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    if is_multiplicative:
        a_ref -= 1
        b_ref -= 1

    return torch.stack([a_ref, b_ref], dim=1)


def apply_g(X, g, p, operation: str):
    """
    Applies a group element g = ('r' or 's', k) to a decoded batch X of shape (n, 2, d).
    """
    g_type, k = g

    if g_type == 'r':
        return r(X, k, p, operation)
    elif g_type == 's':
        return s(X, k, p, operation)
    else:
        raise ValueError(f"Unknown transformation type: {g_type}")

        
def get_G_orbits(X, G, p, operation: str):
    """
    Computes the orbit of a one-hot encoded batch X under a set of group elements G.
    
    Args:
        X (torch.Tensor): one-hot encoded batch of shape (n_samples, 2*p) for non-abelian,
                          (n_samples, 2*prod(p)) for abelian
        G (list): list of group elements, each g = ('r' or 's', k)
        p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)
        
    Returns:
        X_orbit_one_hot (torch.Tensor): one-hot encoded tensor of the orbit, including original X
    """
    device = X.device
    
    # Step 1: Decode data
    if operation == "abelian":
        X_decoded = decode_one_hot(X)
        X_decoded, _ = decode_abelian(X_decoded, None, p)
    else:
        X_decoded = decode_one_hot(X)

    # Step 2: Apply group action
    transformed = [X_decoded]  # include original samples
    for g in G:
        X_g = apply_g(X_decoded, g, p, operation)
        transformed.append(X_g)

    # Step 3: Stack results and remove duplicates
    X_all = torch.cat(transformed, dim=0)
    X_all_unique = torch.unique(X_all, dim=0)

    # Step 4: One-hot encode back
    if operation == "abelian":
        X_orbit, _ = encode_abelian(X_all_unique, None, p)
        X_orbit_one_hot = F.one_hot(X_orbit, num_classes=math.prod(p)).view(len(X_orbit), -1).double()
    else:
        num_classes = p if isinstance(p, int) else p[0]
        X_orbit_one_hot = F.one_hot(X_all_unique, num_classes=num_classes).view(len(X_all_unique), -1).double()

    return X_orbit_one_hot


def theoretical_predictions(X, G, p, operation: str):
    """
    Computes the orbit of a one-hot encoded batch X under G removing the original X (theoretical predictions).
    
    Args:
        X (torch.Tensor): one-hot encoded batch
        G (list): list of group elements
        p (int or list[int]): prime modulus or group sizes
        operation (str): operation type
    """
    is_multiplicative = operation in MULTIPLICATIVE_MODULO_OPERATIONS
    device = X.device
    # Compute the full orbit (includes original samples)
    X_orbit = get_G_orbits(X, G, p, operation)

    # Decode back to integer pairs
    if operation == "abelian":
        X_decoded = decode_one_hot(X)
        X_decoded, _ = decode_abelian(X_decoded, None, p)
        
        X_orbit_decoded = decode_one_hot(X_orbit)
        X_orbit_decoded, _ = decode_abelian(X_orbit_decoded, None, p)
    else:
        X_decoded = decode_one_hot(X)
        X_orbit_decoded = decode_one_hot(X_orbit)

    # Remove original samples
    if operation == "abelian":
        X_decoded_set = set(tuple(x.flatten().tolist()) for x in X_decoded)
        X_new_list = [x for x in X_orbit_decoded if tuple(x.flatten().tolist()) not in X_decoded_set]
        if len(X_new_list) == 0:
            return torch.empty((0, X.shape[1]), device=device, dtype=torch.double)
        
        X_new = torch.stack([x.clone().detach().to(device) for x in X_new_list], dim=0)  # shape [n_new,2,d]
        
    else:
        X_decoded_set = set(map(tuple, X_decoded.tolist()))
        X_new_list = [row for row in X_orbit_decoded.tolist() if tuple(row) not in X_decoded_set]
        if len(X_new_list) == 0:
            return torch.empty((0, X.shape[1]), device=device, dtype=torch.double)
        
        X_new = torch.tensor(X_new_list, device=device)

    # One-hot encode back
    if operation == "abelian":
        X_new_encoded, _ = encode_abelian(X_new, None, p)
        X_preds = F.one_hot(X_new_encoded, num_classes=math.prod(p)).view(len(X_new_encoded), -1).double()
    else:
        num_classes = (p-1) if is_multiplicative else p
        X_preds = F.one_hot(X_new, num_classes=num_classes).view(len(X_new), -1).double()

    return X_preds

def compare_theory(X_te_correct: torch.Tensor, X_preds: torch.Tensor) -> tuple[float, float]:
    """
    Compare theoretical predictions with true correct samples.
    """
    
    # Convert tensors to sets of tuples for comparison
    correct_set = set(map(tuple, X_te_correct.tolist()))
    preds_set   = set(map(tuple, X_preds.tolist()))

    # Compute intersection
    intersection = correct_set & preds_set

    # Metrics
    recall = len(intersection) / len(correct_set) if correct_set else 0.0
    precision = len(intersection) / len(preds_set) if preds_set else 0.0

    return recall, precision

def get_permutation_representation(p, operation, g):
    """
    Compute the permutation matrix for a group element g acting on one-hot concatenated data.

    Args:
        p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)
        g (tuple): ('r' or 's', k)

    Returns:
        M (torch.Tensor): permutation matrix of shape (2*p,2*p) or (2*prod(p),2*prod(p)),
                          for multiplicative operations: 2*(p-1) × 2*(p-1)
    """
    g_type, k = g

    # Determine block size
    if operation in ("x*y", "x/y"):
        block_size = p - 1
    elif operation == "abelian":
        block_size = math.prod(p)
    else:
        block_size = p

    n = 2 * block_size
    M = torch.zeros((n, n), dtype=torch.double)

    # ABELIAN CASE
    if operation == "abelian":
        a_tuples = list(itertools.product(*[range(pi) for pi in p]))
        b_tuples = list(itertools.product(*[range(pi) for pi in p]))
        for i, a in enumerate(a_tuples):
            for j, b in enumerate(b_tuples):
                x = torch.tensor([[a, b]], dtype=torch.long)
                x_encoded, _ = encode_abelian(x, None, p)   # Mixed-radix encoding
                a_idx, b_idx = x_encoded[0]  
                
                # Apply transformation
                if g_type == 'r':
                    x_new = r(x, k, p, operation)[0]  # shape (2,d)
                elif g_type == 's':
                    x_new = s(x, k, p, operation)[0]
                    
                else:
                    raise ValueError(f"Unknown transformation {g_type}")
                
                x_new, _ = encode_abelian(x_new.unsqueeze(0), None, p) # Mixed-radix encoding from (1,2,d) -> (1,2)
                x_new = x_new.squeeze(0) # (1,2) -> (2,)
                
                # Map to permutation matrix indices
                if g_type == 'r':
                    M[x_new[0].item(), a_idx.item()] = 1.0
                    M[block_size + x_new[1].item(), block_size + b_idx.item()] = 1.0
                elif g_type == 's':
                    M[x_new[0].item(), block_size + b_idx.item()] = 1.0
                    M[block_size + x_new[1].item(), a_idx.item()] = 1.0
    
    # MODULAR ARITHMETIC               
    else: 
        for a in range(block_size):
            for b in range(block_size):
                # Create decoded sample
                x = torch.tensor([[a, b]], dtype=torch.long)

                # Apply transformation
                if g_type == 'r':
                    x_new = r(x, k, p, operation)[0]  # shape (2,)
                elif g_type == 's':
                    x_new = s(x, k, p, operation)[0]
                else:
                    raise ValueError(f"Unknown transformation {g_type}")

                # Map to permutation matrix indices
                if g_type == 'r':
                    M[x_new[0].item(), a] = 1.0
                    M[block_size + x_new[1].item(), block_size + b] = 1.0
                elif g_type == 's':
                    M[x_new[0].item(), block_size + b] = 1.0
                    M[block_size + x_new[1].item(), a] = 1.0

    return M

def permutation_representations(p, operation: str):
    """
    Generate all permutation representation matrices for a given group action.

    Args:
        p (int or list[int]): Prime modulus (int) or list of group sizes for "abelian" case
        operation (str): Operation ("x+y", "x*y", "abelian", etc.)

    Returns:
        dict: mapping from group element tuples (('r' or 's'), k) to permutation matrix M
    """
    reps = {}

    # ABELIAN CASE
    if operation == "abelian":
        k_tuples = list(itertools.product(*[range(pi) for pi in p]))

        for k_vec in k_tuples:
            reps[('r', k_vec)] = get_permutation_representation(p, operation, ('r', list(k_vec)))
            reps[('s', k_vec)] = get_permutation_representation(p, operation, ('s', list(k_vec)))
    
    # MODULAR ARITHMETIC
    elif operation in ALL_OPERATIONS:
        # choose range of k depending on operation type
        if operation in MULTIPLICATIVE_MODULO_OPERATIONS:
            k_values = range(1, p)               # multiplicative group: k = 1..p-1
        else:
            k_values = range(p)                  # k = 0..p-1

        # Build matrices
        for k in k_values:
            reps[('r', k)] = get_permutation_representation(p, operation, ('r', k))
            reps[('s', k)] = get_permutation_representation(p, operation, ('s', k))

    else:
        raise ValueError(f"Unsupported operation type: {operation}")

    return reps

def plot_reflection_axis(ax, p, k, operation='x+y', group_sizes = None, **kwargs):
    if operation in ADDITIVE_MODULO_OPERATIONS:
        plot_cyclic_diagonal(ax, p, k, operation, **kwargs)
    elif operation in MULTIPLICATIVE_MODULO_OPERATIONS:
        plot_multiplicative_reflection_axis(ax, p, k, operation, **kwargs)
    elif operation == 'abelian':
        plot_abelian_reflection_axis(ax, group_sizes, k, **kwargs)
    else:
        raise ValueError(f"Unknown operation type: {operation}")

def plot_abelian_reflection_axis(ax, group_sizes, k_tuple, **kwargs):
    """
    Plot reflection axis for an abelian group Z_p1 × Z_p2 × ... × Z_pd
    """

    def encode_single(x):
        """Mixed-radix encoding of tuple x given group_sizes."""
        result = 0
        multiplier = 1
        for xi, pi in zip(reversed(x), reversed(group_sizes)):
            result += xi * multiplier
            multiplier *= pi
        return result

    all_a = list(product(*[range(pi) for pi in group_sizes]))

    xs, ys = [], []
    for a in all_a:
        b = tuple((ai + ki) % pi for ai, ki, pi in zip(a, k_tuple, group_sizes))
        a_idx = encode_single(a)
        b_idx = encode_single(b)
        xs.append(b_idx)
        ys.append(a_idx)

    cell_size = 7200 / (math.prod(group_sizes)**2)
    ax.scatter(xs, ys, s= cell_size, **kwargs)
    
def plot_multiplicative_reflection_axis(ax, p, k, operation='x*y', **kwargs):
    """
    Plot reflection axis for multiplicative operations mod p.
    """
    prime = p + 1
    a = torch.arange(1, prime)
    if operation == "x*y":
        b = (a * k) % prime
    elif operation == "x/y":
        ak = (a * k) % prime
        b = torch.full_like(ak, -1)
        nonzero = ak != 0
        if nonzero.any():
            b[nonzero] = torch.tensor(
                [pow(x.item(), -1, prime) for x in ak[nonzero]],
                dtype=ak.dtype,
                device=ak.device
            )
    else:
        raise ValueError(f"Unknown operation '{operation}', expected 'x*y' or 'x/y'.")

    mask = b != -1
    a = a[mask]
    b = b[mask]
    a = a - 1
    b = b - 1
    cell_size = 7200 / (p**2)
    ax.scatter(b.numpy(), a.numpy(), s=cell_size, **kwargs)
        
def plot_cyclic_diagonal(ax, p, k, operation='x+y', **kwargs):
    """
    Plot a reflection axis on a cyclic p by p grid for addition or subtraction.
    """
    k = - k % p
    x = np.arange(p)

    if operation == 'x+y':
        y = (x + k) % p
        wrap_idx = np.where(np.diff(y) < 0)[0]
    elif operation == 'x-y':
        y = (-x + k) % p
        wrap_idx = np.where(np.diff(y) > 0)[0]
    else:
        raise ValueError(f"Unknown operation '{operation}', expected 'x+y' or 'x-y'.")

    if wrap_idx.size == 0:
        segments = [(x, y)]
    else:
        cut = wrap_idx[0] + 1
        segments = [(x[:cut], y[:cut]), (x[cut:], y[cut:])]

    for xs, ys in segments:
        if len(xs) > 1:
            ax.plot(xs, ys, **kwargs)
        elif len(xs) == 1:
            cell_size = 7200 / (p**2)
            ax.scatter(xs[0], ys[0], s=cell_size, **kwargs)

            
def visualize_ab_grid(X_tr: torch.Tensor, X_te_correct: torch.Tensor, X_te_incorrect: torch.Tensor,
    reflection_axis=None, operation='x+y', colors=None, title="RFM Predictions", group_sizes = None):
    """
    Visualizes training, correct, and incorrect examples on a p by p grid.

    Args:
        X_tr            : One-hot encoded training samples
        X_te_correct    : One-hot encoded correctly predicted test samples
        X_te_incorrect  : One-hot encoded incorrectly predicted test samples
        reflection_axis : (...)
        colors          : dict with optional RGB tuples, e.g.
                          {'train': (0.5, 0.5, 0.5),
                           'correct': (0.0, 1.0, 0.0),
                           'incorrect': (1.0, 0.0, 0.0)}
        title           : title for the plot
    """

    # Default colors
    if colors is None:
        colors = {
            "train": (0.5, 0.5, 0.5),      # grey
            "correct": (0.4, 0.8, 1.0),    # light-blue
            "incorrect": (1.0, 0.0, 0.0),  # red
        }

    assert X_tr.shape[1] % 2 == 0, "Each sample should have 2p columns"
    p = X_tr.shape[1] // 2

    grid = np.ones((p, p, 3))  # white background

    def mark_entries(X, color_rgb):
        if X is None or X.numel() == 0:
            return
        decoded = decode_one_hot(X)  # shape (n, 2)
        for a, b in decoded:
            grid[a.item(), b.item()] = color_rgb

    mark_entries(X_tr, colors["train"])
    mark_entries(X_te_correct, colors["correct"])
    mark_entries(X_te_incorrect, colors["incorrect"])

    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    plt.title(title, fontsize=22)
    plt.xlabel("Second operand", fontsize=18)
    plt.ylabel("First operand", fontsize=18)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    if reflection_axis is not None:
        plot_reflection_axis(plt, p, reflection_axis, operation, color='black', group_sizes = group_sizes, linewidth=3, alpha=0.8)
    plt.show()

    return grid