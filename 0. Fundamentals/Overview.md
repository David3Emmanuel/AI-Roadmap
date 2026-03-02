# Cycle #0 - Backpropagation & Multi-Layer Perceptrons (MLP).

## Implementation Objective

To build a tiny Autograd Engine (automatic differentiation system) from scratch, use it to train a small neural network to solve a logic puzzle (like XOR), and then rewrite the exact same network using PyTorch to see the difference.

## Steps

### 0.1 Scalar Automatic Differentiation

- **Resource:** Andrej Karpathy's video: _"The spelled-out intro to neural networks and backpropagation: building micrograd"_.
- **Implementation:** Scalar autograd engine + forward/reverse mode autodiff.

### 0.2 Vector Automatic Differentiation

To understand how autodiff extends to vectors and matrices.

- **Resource:** _"The Matrix Calculus You Need for Deep Learning"_ Parr & Howard.
- **Implementation:** Vector autograd engine.

### 0.3 XOR Network & PyTorch Rewrite

- Use my autograd engine to train a small MLP to solve XOR.
- Rewrite the exact same network in PyTorch and compare.
