import torch
from torch.nn import Module, Sigmoid
import time


## Inference time only + Assume Autoregressive decoder (Masked self attention)
## APPLICABLE TO ENCODER SELF ATTENTION AS WELL


class AttentionKVCache(Module):
    def __init__(self, W_key, W_query, W_value, emb_dim, output_dim):
        super().__init__()
        self.W_key = W_key
        self.W_query = W_query
        self.W_value = W_value
        self.emb_dim = emb_dim
        self.sigmoid = Sigmoid()
        self.output_dim = output_dim
        self.K_cache = None
        self.V_cache = None

    def __call__(self, x):
        ## Pass one token at a time.
        ## Step1: Calculate k, q,v vectors for current token
        xt = torch.transpose(x, 0, 1)
        # print(f"Shape of xt: {xt.size()}")
        q = torch.matmul(self.W_query, xt)
        k = torch.matmul(self.W_key, xt)
        v = torch.matmul(self.W_value, xt)

        ## Step2: Update K_cache and V_cache
        if self.K_cache is None:
            self.K_cache = k  # k.reshape(-1, 1)
        else:
            self.K_cache = torch.hstack([self.K_cache, k])

        if self.V_cache is None:
            self.V_cache = v  # .reshape(-1, 1)
        else:
            self.V_cache = torch.hstack([self.V_cache, v])

        ## Step 3: Calculate attention
        a = torch.matmul(q.view(1, -1), self.K_cache)
        a = torch.div(a, torch.sqrt(torch.tensor(self.output_dim)))
        a = self.sigmoid(a)

        ## Step 4: Calculate weighted value vector for each position
        z = torch.matmul(self.V_cache, a.view(-1, 1))
        return z


class AttentionNoKVCache(Module):
    def __init__(self, W_key, W_query, W_value, emb_dim, output_dim):
        super().__init__()
        self.W_key = W_key
        self.W_query = W_query
        self.W_value = W_value
        self.emb_dim = emb_dim
        self.sigmoid = Sigmoid()
        self.output_dim = output_dim

    def __call__(self, X):
        ## Pass all tokens till this time step
        # Step 1: Calculate query matrix using X and W_query
        XT = torch.transpose(X, 0, 1)
        Q = torch.matmul(self.W_query, XT)
        K = torch.matmul(self.W_key, XT)
        V = torch.matmul(self.W_value, XT)

        ## Step2: Calculate attention
        A = torch.matmul(torch.transpose(Q, 0, 1), K)
        A = torch.div(A, torch.sqrt(torch.tensor(self.output_dim)))
        A = self.sigmoid(A)

        ## Step 3: Calculate weighted value vector for each position
        Z = torch.matmul(V, torch.transpose(A, 0, 1))

        return Z[:, -1]


if __name__ == "__main__":
    ## We assume single headed attention for simplicity

    emb_dim = 50  ## Dimension of embedding for each token
    output_dim = 15  ## Dimen
    seq_len = 30

    ## Step1: W_key, W_query, W_value are fixed (learned during training)
    ## Assume no bias
    W_key = torch.randn(output_dim, emb_dim)
    W_query = torch.randn(output_dim, emb_dim)
    W_value = torch.randn(output_dim, emb_dim)

    ## Step2: Get X (seq_len, emb_dim)
    X = torch.randn(seq_len, emb_dim)

    no_kv_t0 = time.time()
    attention_no_kv_cache = AttentionNoKVCache(
        W_key, W_query, W_value, emb_dim, output_dim
    )

    torch_list = []
    ## Inference happens one token at a time using a loop
    for i in range(1, seq_len + 1):
        x = X[:i, :]  # Till ith token (ith row)
        zi = attention_no_kv_cache(x).reshape(-1, 1)
        torch_list.append(zi)

    Z_no_KV_cache = torch.hstack(torch_list)
    no_kv_t1 = time.time()
    no_kv_total_time = no_kv_t1 - no_kv_t0
    print(f"Total time for Z_no_KV_cache: {no_kv_total_time}s")
    print(f"Z_no_KV_cache shape: {Z_no_KV_cache.size()}")
    # print(f"Z_no_KV_cache: \n {Z_no_KV_cache}")

    kv_t0 = time.time()
    attention_kv_cache = AttentionKVCache(W_key, W_query, W_value, emb_dim, output_dim)
    kvcache_torch_list = []
    ## Inference happens one token at a time using a loop
    for i in range(1, seq_len + 1):
        x = X[i - 1 : i, :]  # Till ith token (ith row)
        zi = attention_kv_cache(x).reshape(-1, 1)
        kvcache_torch_list.append(zi)
    Z_KV_cache = torch.hstack(kvcache_torch_list)
    kv_t1 = time.time()
    kv_total_time = kv_t1 - kv_t0
    print(f"Total time for Z_KV_cache: {kv_total_time}s")
    print(f"Z_KV_cache shape: {Z_KV_cache.size()}")
    # print(f"Z_KV_cache: \n {Z_KV_cache}")

    print(
        f"Is Z_no_KV_cache and Z_KV_cache equal: {torch.allclose(Z_no_KV_cache, Z_KV_cache, atol=1e-2)}"
    )
    print(
        f"Time proprtion: no_kv_total_time/kv_total_time : {no_kv_total_time/kv_total_time}"
    )
