# Breaking Data Symmetry is Needed For Generalization in Feature Learning Kernels

Code accompanying the paper ["Breaking Data Symmetry is Needed for Generalization in Feature Learning Kernels"]() (to be archived soon).

This repository is built on top of [nmallinar/rfm-grokking](https://github.com/nmallinar/rfm-grokking).

## Installation details

Dependencies can be installed via
```bash
pip install -r requirements.txt
```

## Running the code

+ `main.py` contains the CLI for running experiments
+ `experiments.ipynb` reproduces the main experiments from the paper.

## Files overview

The main files of this repository are:

| File                | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `agop_utils.py`     | Functions for computing AGOP updates.                            |
| `utils.py`          | Utility functions for group actions, plots, and experiments.     |
| `data.py`           | Data generation pipeline.                                        |
| `train_kernel.py`   | Kernel training pipeline using RFM.                              |
| `main.py`           | CLI entry point for experiments.                                 |
| `experiments.ipynb` | Jupyter notebook recreating the main experiments from the paper. |

## Preprocessing

Data samples correspond to the concatenated one-hot encoded group elements. Different groups and operations are encoded differently.

+ Additive operations (`x+y`, `x-y`): Tuple form `(a,b)` -> One-hot form `e_a || e_b`
+ Multiplicative operations (`x*y`, `x/y`): Tuple form `(a,b)` -> Shifted tuple form `(a-1, b-1)` (since 0 is not a valid element) -> One-hot form `e_(a-1) || e_(b-1)`
+ Abelian groups: Tuple form `((a₁, a₂, ...),(b₁, b₂, ...))` -> Mixed radix encoded form `(a,b)` -> One-hot form `e_a || e_b`

For clarity and demonstration, preprocessing is handled per function rather than in a unified pipeline (a production-oriented version could unify preprocessing into a single pipeline for greater modularity). Different functions operate on different data representations.
