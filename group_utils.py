"""
Utility functions for group actions.
"""
import torch
import torch.nn.functional as F
import math
import itertools

torch.set_default_dtype(torch.float64)

from data import (
    encode_abelian,
    decode_abelian,
    decode_one_hot,
    MULTIPLICATIVE_MODULO_OPERATIONS,
    ALL_OPERATIONS,
)

def _r(X, k, p, operation: str):
    """
    Internal implementation of the rotation action r^k, applied to batch X of shape (n, 2, d) where each row is a sample (a, b).
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


def _s(X, k, p, operation: str):
    """
    Internal implementation of the reflection action sr^k applied to batch X of shape (n, 2, d), where each row is a sample (a, b).
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


def _apply_g(X, g, p, operation: str):
    """
    Internal dispatcher for group actions. Applies a group element g = ('r' or 's', k) to a decoded batch X of shape (n, 2, d).
    """
    g_type, k = g

    if g_type == 'r':
        return _r(X, k, p, operation)
    elif g_type == 's':
        return _s(X, k, p, operation)
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
        X_g = _apply_g(X_decoded, g, p, operation)
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
                    x_new = _r(x, k, p, operation)[0]  # shape (2,d)
                elif g_type == 's':
                    x_new = _s(x, k, p, operation)[0]
                    
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
                    x_new = _r(x, k, p, operation)[0]  # shape (2,)
                elif g_type == 's':
                    x_new = _s(x, k, p, operation)[0]
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