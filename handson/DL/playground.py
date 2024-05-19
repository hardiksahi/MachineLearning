import torch
from torch.nn import LSTM, Dropout

if __name__ == "__main__":
    tensor1 = torch.randn(20, 16)
    print(f"Shape of tensor1: {tensor1.size()}")

    dropout_layer = Dropout(0.2)
    drop = dropout_layer(tensor1)
    print(drop)
