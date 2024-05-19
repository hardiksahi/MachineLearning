import numpy as np
import torch
from torch.nn import BatchNorm1d, LayerNorm

if __name__ == "__main__":
    X = np.random.rand(3, 4)  ## Batch_size = 3
    W = np.ones((2, 4))
    b = np.ones((2, 1))
    out = np.dot(W, np.transpose(X)) + b  ## (2(features/ output neurons),3(batch_size))
    batch_mean = np.mean(out, axis=1).reshape(-1, 1)
    batch_variance = np.var(out, axis=1).reshape(-1, 1)
    batch_norm_out = (out - batch_mean) / np.sqrt(batch_variance + 1e-5)
    print(f"[BatchNorm Numpy based] batch_norm_out:\n {batch_norm_out}")

    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_W = torch.tensor(W, dtype=torch.float32)
    tensor_b = torch.tensor(b, dtype=torch.float32)
    tensor_out = torch.matmul(tensor_W, torch.transpose(tensor_X, 0, 1)) + tensor_b
    tensor_mu = torch.mean(tensor_out, dim=1, keepdim=True)
    tensor_var = torch.var(tensor_out, dim=1, keepdim=True, unbiased=False)
    tensor_sigma = torch.sqrt(tensor_var + 1e-5)
    tensor_batch_norm_out = (tensor_out - tensor_mu) / tensor_sigma
    print(f"[BatchNorm Pytorch manual based]:\n {tensor_batch_norm_out}")

    batch_norm_layer = BatchNorm1d(
        2, affine=False
    )  ## 2 is number of features/ neurons in hidden layer
    layer_batch_norm = batch_norm_layer(
        torch.transpose(tensor_out, 0, 1)
    )  ## BatchNorm1D layer needs dimensions (= number of neurons in hidden layer) along columns
    print(
        f"[BatchNorm Pytorch BatchNorm1D based]:\n {torch.transpose(layer_batch_norm, 0, 1)}"
    )

    print("***********************************")

    layer_mean = np.mean(out, axis=0).reshape(1, -1)
    layer_variance = np.var(out, axis=0).reshape(1, -1)
    layer_norm_out = (out - layer_mean) / np.sqrt(layer_variance + 1e-5)
    print(f"[LayerNorm Numpy based] layer_norm_out:\n {layer_norm_out}")

    layer_tensor_mu = torch.mean(tensor_out, dim=0, keepdim=True)
    layer_tensor_var = torch.var(tensor_out, dim=0, keepdim=True, unbiased=False)
    layer_tensor_sigma = torch.sqrt(layer_tensor_var + 1e-5)
    layer_norm_tensor_out = (tensor_out - layer_tensor_mu) / layer_tensor_sigma
    print(f"[LayerNorm Pytorch manual based]:\n {layer_norm_tensor_out}")

    layer_norm_layer = LayerNorm(
        2
    )  ## 2 is number of features/ dimensions/ neurons in hidden layer
    layer_layer_norm = layer_norm_layer(torch.transpose(tensor_out, 0, 1))
    print(
        f"[LayerNorm Pytorch LayerNorm based]:\n {torch.transpose(layer_layer_norm, 0, 1)}"
    )
