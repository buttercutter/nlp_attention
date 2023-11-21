import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USE_COVARIANCE_CHECK = 1
USE_VARIANCE_CHECK = not USE_COVARIANCE_CHECK

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
            print("using covariance as feedback")
            self.self_attn = CovarianceTracker(ShapedSelfAttention(d_model, nhead), batch_size)
            self.linear1 = CovarianceTracker(nn.Linear(d_model, dim_feedforward), batch_size)
            self.linear2 = CovarianceTracker(nn.Linear(dim_feedforward, d_model), batch_size)
            self.linear1.out_features = dim_feedforward
            self.linear2.out_features = d_model

        elif USE_VARIANCE_CHECK:
            print("using variance as feedback")
            self.self_attn = BatchNormTracker(ShapedSelfAttention(d_model, nhead), batch_size)
            self.linear1 = BatchNormTracker(nn.Linear(d_model, dim_feedforward), batch_size)
            self.linear2 = BatchNormTracker(nn.Linear(dim_feedforward, d_model), batch_size)
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


"""
The covariance information might be used to implement some form of normalization or regularization,
or to perform other transformations based on the covariance.

For example, it could be used to implement a form of "whitening"
where the activations are transformed to have zero mean and unit covariance.

This can help to improve the conditioning of the optimization problem and can lead to improved model performance.


    If the data at different positions in the sequence are expected to have different statistical properties, it would be more appropriate to keep separate means for each position in the sequence. This would be the case, for example, in certain natural language processing tasks where the position in the sentence can have a significant impact on the meaning of a word. In this case, you would use batch_mean = x.mean(dim=0) and self.register_buffer('running_mean', torch.zeros(x.size(1), self.layer.out_features))

    Conversely, if the data at different positions in the sequence are interchangeable or have similar statistical properties, it would make more sense to calculate the mean across the entire batch and sequence length. This might be the case, for example, in certain image processing tasks where the position of a pixel in the image does not significantly affect its value. In this case, you would use batch_mean = x.mean(dim=[0, 1]) and self.register_buffer('running_mean', torch.zeros(layer.out_features))


The self.register_buffer('running_covariance', torch.eye(layer.out_features, device=device)) is used to initialize the running covariance as an identity matrix rather than a matrix of all ones. An identity matrix is a square matrix in which all the elements of the principal (main) diagonal are ones and all other elements are zeros.

This is done because the covariance matrix is a measure of how much each of the dimensions of the data vary and are correlated with each other. An identity matrix is the simplest form of a covariance matrix, where all variables are uncorrelated and have a variance of one. It is used as an initial value for the running covariance for the reasons below:

    Uncorrelated Features: In an identity matrix, all off-diagonal elements are zero, implying that the variables (or features in this case) are uncorrelated, which is a good assumption to start with in the absence of any other information about the correlations between features.

    Unit Variance: The diagonal elements of a covariance matrix represent the variance of the variables. By setting these elements to one in the identity matrix, we're effectively normalizing the features to have a unit variance, which is a common practice in machine learning to ensure that all features contribute equally to the model.

In contrast, initializing the covariance matrix to all ones would imply that all features are perfectly correlated with each other and have the same variance, which might not be a good assumption for most datasets.

Credit: GPT-4 AI chatbot
"""
class CovarianceTracker(nn.Module):
    def __init__(self, layer, epsilon=1e-5, alpha=0.99):
        super().__init__()
        self.layer = layer
        self.epsilon = epsilon  # small constant to prevent division by zero
        self.alpha = alpha  # decay rate for the running averages

        #self.register_buffer('running_mean', torch.zeros(layer.out_features, device=device))
        self.register_buffer('running_covariance', torch.eye(layer.out_features, device=device))

    def forward(self, x):
        x = self.layer(x)

        # `running_mean` has a shape of (seq_len, d_model)
        self.register_buffer('running_mean', torch.zeros(x.size(1), self.layer.out_features, device=device))

        batch_mean = x.mean(dim=0)
        #batch_mean = x.mean(dim=[0, 1])

        # we subtract the mean from x for covariance calculation
        x_centered = x - batch_mean.unsqueeze(0)
        batch_covariance = x_centered.transpose(-2, -1).matmul(x_centered) / x_centered.size(0)
        batch_covariance = batch_covariance.mean(dim=0)

        # update the running mean and covariance with exponential moving average
        self.running_mean.mul_(self.alpha).add_(batch_mean.data, alpha=1-self.alpha)
        self.running_covariance.mul_(self.alpha).add_(batch_covariance.data, alpha=1-self.alpha)

        # compute the whitened activations
        x_whitened = (x - self.running_mean) @ torch.linalg.inv(torch.sqrt(self.running_covariance + self.epsilon))

        return x_whitened


"""
Whitening can be a computationally expensive operation, especially for high-dimensional data,
due to the need to compute the inverse square root of the covariance matrix,
see the logic for `x_whitened` variable inside `CovarianceTracker` class.

Computing the inverse of a matrix has a time complexity of O(n^3) for a naive algorithm (Gaussian elimination),
and even with more advanced methods (like the Strassen algorithm or Coppersmith–Winograd algorithm),
it can still be quite expensive for large matrices.

Hence the birth of `BatchNormTracker` class,
but we still need to prove that using Variance instead of Covariance as feedback would perform similarly
according to equation (26) inside Theorem 4.2 of the Shaped Transformer paper.

https://kexue.fm/archives/9812#正态分布 and https://kexue.fm/archives/8823#新的因子 are really confusing.
I would need to relate these two articles back to equation (26)
"""
class BatchNormTracker(nn.Module):
    def __init__(self, layer, epsilon=1e-5, momentum=0.1):
        super().__init__()
        self.layer = layer
        self.epsilon = epsilon
        self.momentum = momentum

        #self.register_buffer('running_mean', torch.zeros(layer.out_features, device=device))
        self.register_buffer('running_var', torch.ones(layer.out_features, device=device))

    def forward(self, x):
        x = self.layer(x)

        # `running_mean` has a shape of (seq_len, d_model)
        self.register_buffer('running_mean', torch.zeros(x.size(1), self.layer.out_features, device=device))

        if self.training:
            batch_mean = x.mean(dim=0)
            #batch_mean = x.mean(dim=[0, 1])

            batch_var = x.var(dim=[0, 1], unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean.data * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(batch_var.data * self.momentum)
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.epsilon)
        return x_norm


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
