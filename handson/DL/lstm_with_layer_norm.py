import pandas as pd
import numpy as np
import torch
from torch.nn import Module, LSTM, LayerNorm

if __name__ == "__main__":
    rnn = LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
    input = torch.randn(5, 3, 10)
    output, (hn, cn) = rnn(input)
    print(f"Shape of output: {output.size()}")
    print(f"Shape of hn: {hn.size()}")
    print(f"Shape of cn: {cn.size()}")

    layer_norm = LayerNorm(20)
    ln_output = layer_norm(output)
    print(f"Shape of ln_output: {ln_output.size()}")

    ## Calculate on own
    mean = torch.mean(output, dim=-1, keepdim=True)
    print(f"Shape of mean: {mean.size()}")

    variance = torch.var(output, dim=-1, keepdim=True, unbiased=False)
    print(f"Shape of variance: {variance.size()}")

    sigma = torch.sqrt(variance + 1e-5)
    print(f"Shape of sigma: {sigma.size()}")

    layer_norm_manual = (output - mean) / sigma
    print(
        f"Is output of LayerNorm and manual calcualtion equal: {torch.allclose(ln_output, layer_norm_manual)}"
    )
