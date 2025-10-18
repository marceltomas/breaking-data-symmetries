import argparse
import torch
import json
from easydict import EasyDict

from train_kernel import train

def parse_args():
    parser = argparse.ArgumentParser(description="Run RFM training with configurable options.")

    # --- Data args ---
    parser.add_argument("--operation", type=str, default="x+y", help="Operation type: 'x+y', 'x-y', 'x*y', 'x/y' or 'abelian'")
    parser.add_argument("--prime", type=int, nargs="+", default=[29], help="Prime modulus (int) or list of group sizes for 'abelian' case")
    parser.add_argument("--partition_type", type=str, default="random", choices=["random", "degenerate"],
                        help="Type of data partition: 'random' or 'degenerate'")
    parser.add_argument("--training_fraction", type=float, default=0.5, help="Fraction of data for training (random partition only)")

    parser.add_argument("--reflections", type=str, default="[0]", help="Reflection indices k (degenerate partition only)")
    parser.add_argument("--n_test_to_train", type=int, default=0, help="# points to move from test to train (degenerate partition only)")
    parser.add_argument("--n_train_to_test", type=int, default=0, help="# points to move from train to test (degenerate partition only)")
    parser.add_argument("--n_pairs_to_test", type=int, default=0, help="# reflected pairs to move from train to test (degenerate partition only)")

    parser.add_argument("--M_reflection", type=int, default=None, help="Reflection index k to initialize M from (or None)")

    # --- RFM args ---
    parser.add_argument("--iters", type=int, default=50, help="# of RFM iterations")
    parser.add_argument("--ridge", type=float, default=0.0, help="Regularization parameter for kernel regression")
    parser.add_argument("--bandwidth", type=float, default=2.5, help="Bandwidth for Gaussian kernel")
    parser.add_argument("--kernel_type", type=str, default="gaussian", choices=["gaussian", "quadratic"], help="Kernel type")
    parser.add_argument("--print_progress", action="store_true", help="Whether to print progress per iteration")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: 'cuda' or 'cpu'")

    args = parser.parse_args()

    if args.operation != "abelian" and len(args.prime) == 1:
        args.prime = args.prime[0]  # Convert [p] → p for non-abelian operations
        
    try:
        args.reflections = json.loads(args.reflections)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid format for --reflections: {args.reflections}. Use e.g. '[2,3]' or '[[2,3],[1,2]]'")
        
    return args


def main():
    args = parse_args()

    # Wrap in EasyDicts
    data_args = EasyDict({
        "operation": args.operation,
        "prime": args.prime,
        "partition_type": args.partition_type,
        "training_fraction": args.training_fraction,
        "reflections": args.reflections,
        "n_test_to_train": args.n_test_to_train,
        "n_train_to_test": args.n_train_to_test,
        "n_pairs_to_test": args.n_pairs_to_test,
        "M_reflection": args.M_reflection
    })

    rfm_args = EasyDict({
        "iters": args.iters,
        "ridge": args.ridge,
        "bandwidth": args.bandwidth,
        "ntk_depth": 2,
        "kernel_type": args.kernel_type,
        "print_progress": args.print_progress,
        "device": torch.device(args.device)
    })

    results, rfm_args = train(data_args, rfm_args)

    # Print summary
    print("\n--- Final Test Accuracy ---")
    print(f"{results['test_accs'][-1]:.4f}")
    print("--- Final Train Accuracy ---")
    print(f"{results['train_accs'][-1]:.4f}")


if __name__ == "__main__":
    main()