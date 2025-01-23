# Power Attention

A CUDA implementation of symmetric power attention, achieving transformer-level performance with linear-cost RNN computation.

## Overview

Power Attention implements symmetric power transformers as described in [Buckman et al. (2024)](https://manifestai.com/articles/symmetric-power-transformers/). The key insight is replacing the traditional softmax mechanism with symmetric power embeddings, which enables:

1. **Linear Complexity**: O(n) computation and memory through an RNN formulation
2. **Tractable State Size**: For a 124M parameter model with p=4, the state size is ~14GB
3. **Theoretical Guarantees**: Proven equivalence to power-based attention mechanisms

## Mathematical Foundation

The architecture replaces the softmax in traditional transformers with an even power:

$$
Y_i = \sum_{j=1}^i A_{ij} V_j \qquad A_{ij} = \frac{ \phi(Q_i)^T \phi(K_j)}{\sum_{k=1}^i \phi(Q_i)^T \phi(K_k) }
$$

where $\phi$ is the symmetric power embedding function. This formulation enables an equivalent RNN computation:

$$
Y_{i} = \frac{S_i \phi(Q_i)}{Z_i \phi(Q_i)} \qquad Z_i = Z_{i-1} + \phi(K_i)^T \qquad S_i = S_{i-1} + V_i \phi(K_i)^T
$$

In practice, the normalization above can be replaced by a layernorm layer without sacrificing any performance, so this package implements the following mathematical operation:

$$
Y_i = \text{Norm}(\sum_{j=1}^{i}\phi(Q_i)^T \phi(K_j)V_j) 
$$

with the corresponding recurrent computation:

$$
Y_i = \text{Norm}(S_i \phi(Q_i)) \qquad S_i = S_{i-1} + V_i \phi(K_i)^T
$$

where $\text{Norm}$ denotes a layernorm layer (Ba, Kiros, and Hinton 2016).

For a detailed derivation of the symmetric power embedding $\phi$ and its properties, please refer to the [mathematical background](https://manifestai.com/articles/symmetric-power-transformers/#4-1-mathematical-background) section in the paper.

## Key Features

- **Linear-Cost Attention**: $O(n)$ computation and memory complexity through RNN formulation
- **Symmetric Power Embedding**: Novel embedding function based on symmetric tensors
- **Configurable State Size**: Control state size vs accuracy tradeoff with power parameter p (2-4)
- **Rotary Compatibility**: Full compatibility with rotary embeddings, unlike other linear transformer variants

## Getting Started

- [Installation](installation.md): Build configuration and requirements
- [Quickstart](quickstart.md): API usage and PyTorch integration
- [Benchmarking](benchmarking.md): Performance evaluation methodology