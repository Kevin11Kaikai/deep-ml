import numpy as np

def softmax(X, axis=-1): # numerically stable softmax
    shifted_X = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(shifted_X)
    return exp_X/np.sum(exp_X, axis=axis, keepdims=True)
def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    x_new = x_new.reshape(1,-1) # (d_model,)->(1,d_model)
    # 1. QKV Projection
    Q = x_new @ W_Q # (1,d_model) @ (d_model,d_k)->(1,d_k)
    K = x_new @ W_K
    V = x_new @ W_V
    # 2. append k,v as the new rows of the cache
    if cache is None:
        K_cache, V_cache = None, None
    else:
        K_cache, V_cache = cache
    if K_cache is None and V_cache is None:
        K_all, V_all = K, V
    else:
        K_all = np.concatenate([K_cache, K],axis=0)
        V_all = np.concatenate([V_cache, V],axis=0)
    # 3. scores
    d_k = Q.shape[-1]
    scores = Q @ K_all.swapaxes(-2,-1) / np.sqrt(d_k)
    weights = softmax(scores)
    # 4. output
    output = weights @ V_all
    d_v = V_all.shape[-1]
    output = output.reshape(d_v)
    return output, (K_all, V_all)
    #pass