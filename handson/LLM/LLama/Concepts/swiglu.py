import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from functools import partial
import torch
from torch.nn import Module, Linear, Sigmoid


## SwiGLU is combination by Swish + GLU
## SwiGLU is a drop ion replacement of linear layer
class GLU(Module):
    ## gated Linear Unit => Uses Sigmnoid as activation
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.lin_layer1 = Linear(in_features=input_dim, out_features=out_dim, bias=True)
        self.lin_layer2 = Linear(in_features=input_dim, out_features=out_dim, bias=True)
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        out1 = self.lin_layer1(x)
        out2 = self.lin_layer2(x)
        out2 = self.sigmoid(out2)
        final = out1 * out2
        return final


class SwiGLU(Module):
    ## Swish Gated Linear Unit => Uses SwishBeta function as activation
    # def swishBeta()
    def __init__(self, input_dim, out_dim, beta):
        super().__init__()
        self.beta = beta
        self.lin_layer1 = Linear(in_features=input_dim, out_features=out_dim, bias=True)
        self.lin_layer2 = Linear(in_features=input_dim, out_features=out_dim, bias=True)

    def __call__(self, x):
        out1 = self.lin_layer1(x)
        out2 = self.lin_layer2(x)

        ## Apply SwishBeta function on out2
        betax = -1 * self.beta * out2
        inverse = 1 / (1 + torch.exp(betax))
        final = out2 * inverse

        return out1 * final


## Swish is smooth approximation of ReLU activation fuunction (max(0,x))
## This script plots multiple activation functions
def plot_function(func_dict):
    ## Restrict the domain (-10 to 10)
    # function_output_map = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(-10, 10, 0.01)

    activation_df = pd.DataFrame()
    for name, function in func_dict.items():
        # function_output_map[name] = [function(element) for element in x]
        activation = [function(element) for element in x]
        function_df = pd.DataFrame({"x": x, "activation": activation, "name": name})
        activation_df = pd.concat([activation_df, function_df], axis=0)

    sns.lineplot(data=activation_df, x="x", y="activation", hue="name", ax=ax)

    return fig


if __name__ == "__main__":
    relu = lambda x: max(0, x)  ## Non differetiable near 0
    gelu = lambda x: x * norm.cdf(x)  ## Diferentiable near 0
    swishBeta = lambda x, beta: x * (1 / (1 + np.exp(-beta * x)))
    activation_fig = plot_function(
        {
            "relu": relu,
            "gelu": gelu,
            "swish1": partial(swishBeta, beta=1),
            "swish0.5": partial(swishBeta, beta=0.5),
        }
    )
    activation_fig.savefig("Activation_Functions.png")

    X = torch.randn(5, 10)
    glu = GLU(input_dim=10, out_dim=3)
    glu_out = glu(X)
    print(f"SHape of glu_out: {glu_out.size()}")

    swiglu = SwiGLU(input_dim=10, out_dim=3, beta=1)
    swiglu_out = swiglu(X)
    print(f"SHape of swiglu_out: {swiglu_out.size()}")

    print(f"glu_out\n: {glu_out}")
    print(f"swiglu_out\n: {swiglu_out}")
