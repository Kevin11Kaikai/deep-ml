import numpy as np

def softmax(X, axis=-1):
    shifted_X = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(shifted_X)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)
def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    X, W_q_lin, W_k_lin, W_v_lin = np.asarray(X), np.asarray(W_q_lin), np.asarray(W_k_lin), np.asarray(W_v_lin)
    # 1. QKV Projection
    Q = X @ W_q_lin.swapaxes(-2,-1)
    K = X @ W_k_lin.swapaxes(-2,-1)
    V = X @ W_v_lin.swapaxes(-2,-1)
    # 2. Scaled Scores
    d_out = Q.shape[-1]
    seq_len = Q.shape[-2]
    scores = Q @ K.swapaxes(-2,-1) / np.sqrt(d_out)
    mask = np.tril(np.ones((seq_len, seq_len), dtype = bool))
    masked_scores = np.where(mask, scores, -np.inf)
    # 3. Weights
    weights = softmax(masked_scores)
    # 4. Outputs
    Output = weights @ V
    return Output.tolist()
    # Your code here
    #pass
