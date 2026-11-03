import numpy as np
from typing import Tuple

def softmax(X, axis=-1):
    # numerically stable softmax
    shifted_X = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(shifted_X)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)
def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return Q,K,V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2,-1) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V

    # Your code here
    #pass


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    # 1. initialization
    seq_len, d_model = Q.shape[-2], Q.shape[-1]
    d_heads = d_model // n_heads
    # 2. reshape Q,K, V
    Q = Q.reshape(seq_len,n_heads,d_heads).swapaxes(-3,-2)
    K = K.reshape(seq_len,n_heads,d_heads).swapaxes(-3,-2)
    V = V.reshape(seq_len,n_heads,d_heads).swapaxes(-3,-2)
    # 3. scores 
    scores = Q @ K.swapaxes(-2,-1) / np.sqrt(d_heads)
    # 4. weights
    weights = softmax(scores)
    # 5. results
    results = weights @ V
    results = results.swapaxes(-3,-2).reshape(seq_len, d_model)
    #results = results.reshape((seq_len, d_heads))
    return results
    # pass