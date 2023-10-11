from typing import Optional, Tuple, Union

import random
import torch
from torch import nn
from torch.nn import functional as F


def _gqa_attn(query, key, value, attention_mask=None, scale_attn_weights=False,
              causal_mask_flag=False, dropout=0.0, local_window_size=None, sink_tokens=1):
    """Group Query Attention implementation."""

    # Check for potential issues before moving on
    if not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f"Expected query, key, and value to be 4-dimensional, but got shapes "
                         f"{query.shape}, {key.shape}, and {value.shape}.")

    print(f"query_len = {query.shape[2]}, key_len = {key.shape[2]}, value_len = {value.shape[2]}")


    """
    Expected shapes: (batch_size, num_heads, query_len, query_dim) similar to _upcast_and_reordered_attn
    """
    batch_size, num_heads, query_len, query_dim = query.shape


    scale_factor = 1.0
    if scale_attn_weights:
        scale_factor /= float(value.size(-1)) ** 0.5

    # if self.scale_attn_by_inverse_layer_idx:
    #         attn_weights = attn_weights / float(self.layer_idx + 1)

    '''
    Scaling the query
    For now we have scale 1.0
    The scale factor has not been integrated into the attention function yet.
    '''

    query = query / scale_factor

    '''
    Determine the number of groups
    For example lets say we have 4 queries heads and 2 keys heads, then we have 2 groups
    '''

    n_groups = query.size(1) // key.size(1)

    if n_groups > 1:
        query_shape = query.shape
        '''
        Lets say the number of group are 2 and head are 2,
        then reshape the query tensor to (batch_size, (2, 2), query_len, query_dim)
        '''
        grouped_shape = (query_shape[0], n_groups, query_shape[1]//n_groups, query_shape[2], query_shape[3])
        query_grouped = query.reshape(grouped_shape)

        '''
        query shape (batch_size, num_groups, num_heads, query_len, query_dim)
        '''
        attn_weights_grouped = torch.matmul(query_grouped, key.transpose(-2, -1))

        '''
        attention_weights_grouped shape (batch_size, num_groups, num_heads, query_len, key_len).
        '''
        attn_weights = attn_weights_grouped.sum(dim=1)

        '''
        attention weights shape: (batch_size, num_heads, query_len, key_len)
        '''

        #print("attn_weights:", attn_weights.shape)

    else:
        '''
        If the number of groups is 1, then we can use the normal attention function
        '''
        attn_weights = torch.matmul(query, key.transpose(-2, -1))


    # Incorporate local attention
    if local_window_size is not None:
        max_seq_len = query.size(-2)
        indices = torch.arange(max_seq_len).to(query.device)
        expanded_indices = indices.unsqueeze(-1).expand(max_seq_len, max_seq_len)
        distance_matrix = torch.abs(expanded_indices - indices.unsqueeze(0))
        print(f"distance_matrix = {distance_matrix}")
        attn_weights.masked_fill_(distance_matrix > local_window_size, float('-inf'))
        print(f"attn_weights AAA = {attn_weights}")


    if attention_mask is not None:
        # Apply the attention mask
        '''
        Input attention_mask shape: (batch_size, num_heads, query_len, key_len)
        '''
        print(f"attention_mask.shape = {attention_mask.shape}")
        print(f"attn_weights.shape = {attn_weights.shape}")
        attn_weights += attention_mask.unsqueeze(1)  # Unsqueeze to Add head dimension


    # Causal masking ensures that the attention mechanism doesn't attend to "future" tokens in sequences.
    if causal_mask_flag:
        causal_mask = torch.ones((query.size(0), query.size(2), key.size(2)), device=query.device, dtype=torch.bool).tril_()
        # causal mask is lower traingular matrix with 1s on the lower triangle and 0s on the upper triangle
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        # print("mask_value:", mask_value)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        # print("attn_weights:", attn_weights)

    # Softmax normalization to get the attention scores
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)


    # Apply dropout if specified
    if dropout > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout)

    # Compute the output by multiplying the attention scores with the value tensor.
    attn_output = torch.matmul(attn_weights, value)


    if sink_tokens > 0:
        sink_query = query[:, :, :sink_tokens, :]

        if n_groups > 1:
            sink_query = sink_query.mean(dim=1)  # or sink_query.sum(dim=1)

        sink_key = key[:, :, :sink_tokens, :]
        sink_value = value[:, :, :sink_tokens, :]

        # Compute the sink attention
        print(f"sink_query.shape = {sink_query.shape}")
        print(f"sink_key.transpose(-2, -1).shape = {sink_key.transpose(-2, -1).shape}")
        sink_attn = nn.functional.softmax(sink_query @ sink_key.transpose(-2, -1) / scale_factor, dim=-1)  # Shape: [B, num_sinks, L]
        print(f"sink_attn.shape = {sink_attn.shape}")
        print(f"sink_value.shape = {sink_value.shape}")
        sink_output = sink_attn @ sink_value  # Shape: [B, num_sinks, d_model]

        # Add padding to sink_output
        pad_len = attn_output.size(2) - sink_output.size(2)
        sink_output = F.pad(sink_output, (0, 0, 0, pad_len))

        # Concatenate the sink output and the main output
        print(f"sink_output.shape = {sink_output.shape}")
        print(f"attn_output.shape = {attn_output.shape}")
        attn_output = torch.cat([sink_output, attn_output], dim=1)  # Shape: [B, L+num_sinks, d_model]


    return attn_output, attn_weights


if __name__ == '__main__':
    # Setting random seed for reproducibility
    torch.manual_seed(42)



    ## Uncomment here to test when num_groups > 1
    # Example tensor shapes
    batch_size = 2
    query_num_heads = 4
    key_value_num_heads = 2
    query_len = 6
    key_len = 6
    dim = 5

    # Generate example tensors
    query = torch.randn(batch_size, query_num_heads, query_len, dim)
    key = torch.randn(batch_size, key_value_num_heads, key_len, dim)
    value = torch.randn(batch_size, key_value_num_heads, key_len, dim)


    # Set number of sink tokens
    # For StreamingLLM, see paper : Efficient Streaming Language Models with Attention Sinks
    # http://arxiv.org/abs/2309.17453
    sink_tokens = 2


    shape = (batch_size, query_len, query_len)
    attention_mask = torch.zeros(shape)

    # Example attention mask, 1s indicate positions we want to include, -inf (or very large negative numbers) indicate positions to exclude
    indices = torch.randperm(attention_mask.numel())  # Assign value of -1e9 randomly
    attention_mask.view(-1)[indices[:5]] = -1e9


    # Run the function
    attn_output, attn_weights = _gqa_attn(query, key, value, attention_mask, scale_attn_weights=True, causal_mask_flag=True, dropout=0.0, local_window_size=3, sink_tokens=sink_tokens)

    print("Attention Output:", attn_output.shape)
    print("Attention Weights:", attn_weights.shape)


    # Print attn weights
    print(f"attn_weights = {attn_weights}")


    # Slice out sink token weights
    sink_attn = attn_weights[:, :, :sink_tokens, :]
    print(f"sink_attn = {sink_attn}")

    # Slice out main attn weights
    main_attn = attn_weights[:, :, sink_tokens:, :]
    print(f"main_attn = {main_attn}")



    # if __name__ == '__main__':
    # Setting random seed for reproducibility

    '''
    ## Uncomment here to test when num_groups = 1
    '''
    # torch.manual_seed(42)

    # # Example tensor shapes
    # batch_size = 2
    # query_num_heads = 2  # <-- Change this to 2
    # key_value_num_heads = 2
    # query_len = 3
    # key_len = 3
    # dim = 5

    # # Generate example tensors
    # query = torch.randn(batch_size, query_num_heads, query_len, dim)
    # key = torch.randn(batch_size, key_value_num_heads, key_len, dim)
    # value = torch.randn(batch_size, key_value_num_heads, key_len, dim)

    # # Example attention mask, 1s indicate positions we want to include, -inf (or very large negative numbers) indicate positions to exclude
    # attention_mask = torch.tensor([
    #     [[0., -1e9, -1e9], [0., 0., -1e9], [0., 0., 0.]],
    #     [[0., -1e9, -1e9], [0., 0., -1e9], [0., 0., 0.]]
    # ])

    # # Run the function
    # attn_output, attn_weights = _gqa_attn(query, key, value, attention_mask, scale_attn_weights=True, causal_mask_flag=True, dropout=0.1)

    # print("Attention Output:", attn_output.shape)
    # print("Attention Weights:", attn_weights.shape)





