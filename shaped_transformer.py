import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USE_COVARIANCE_CHECK = 1


class ShapedReLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.slope_1 = 1.0  # please change this to learnable param during training phase
        self.slope_2 = 1.0  # please change this to learnable param during training phase

    def forward(self, x):
        sqrt_n = np.sqrt(self.width)
        return 1 + self.slope_1 * (1/sqrt_n) * torch.clamp(x, min=0) + \
                   self.slope_2 * (1 - 1/sqrt_n) * torch.clamp(x, max=0)


# See section 4.2 as well as eq(11) and eq(33) for overview of [Shaped Transformer](http://arxiv.org/abs/2306.17759)
class ShapedSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.out_features = d_model
        self.gamma_1 = 1.0  # please change this to learnable param during training phase
        self.gamma_2 = 1.0  # please change this to learnable param during training phase

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        # Scale by width-dependent temperature parameter
        attn_weights = attn_weights / np.sqrt(self.d_model)
        attn_weights = self.softmax(attn_weights)

        # Add identity matrix
        identity = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
        attn_weights = self.gamma_1 * identity + attn_weights

        # Subtract mean
        mean = attn_weights.mean(dim=-1, keepdim=True)
        attn_weights = attn_weights - self.gamma_2 * mean

        # modify attention weights here as necessary for Shaped Transformer methodology
        attn_output = torch.bmm(attn_weights, x)  # apply attention weights directly to x

        return attn_output

class ShapedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()

        if USE_COVARIANCE_CHECK:
            self.self_attn = CovarianceTracker(ShapedSelfAttention(d_model, nhead), batch_size)
            self.linear1 = CovarianceTracker(nn.Linear(d_model, dim_feedforward), batch_size)
            self.linear2 = CovarianceTracker(nn.Linear(dim_feedforward, d_model), batch_size)
            self.linear1.out_features = dim_feedforward
            self.linear2.out_features = d_model
        else:
            self.self_attn = ShapedSelfAttention(d_model, nhead)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ShapedReLU(dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class CovarianceTracker(nn.Module):
    def __init__(self, layer, batch_size):
        super().__init__()
        self.layer = layer
        self.register_buffer('running_covariance', torch.zeros((batch_size, layer.out_features, layer.out_features)))

    def forward(self, x):
        x = self.layer(x)
        centered_activations = x - x.mean(dim=0)
        covariance = torch.einsum('bij,bik->bjk', centered_activations, centered_activations) / centered_activations.size(0)
        self.running_covariance.mul_(0.99).add_(covariance, alpha=0.01)  # Exponential moving average
        return x


###### Testing code ######

if __name__ == '__main__':
    # Setting random seed for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    batch_size = 32
    seq_len = 10
    d_model = 512
    nhead = 8
    dim_feedforward = 2048

    # Generate random data
    data = torch.randn(batch_size, seq_len, d_model).to(device)

    # Define model
    model = ShapedTransformerLayer(d_model, nhead, dim_feedforward).to(device)

    # Forward pass
    output = model(data)

    # Print output shape
    # it should be the same as the input shape, which is (batch_size, seq_len, d_model)
    print(output.shape)
