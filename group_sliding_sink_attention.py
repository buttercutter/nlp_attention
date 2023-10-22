from typing import Optional, Tuple, Union

import numpy
import random
import torch
from torch import nn
from torch.nn import functional as F

import faiss # Added for FOT


def _fot_mem_attn(query, memory_keys, memory_values, topk):
    """
    FOT memory attention implementation
    Focused Transformer: Contrastive Training for Context Scaling
    https://arxiv.org/abs/2307.03170
    Credit: AI chatbot
    """

    # Fetch top k keys using FAISS
    index = faiss.IndexFlatIP(memory_keys.shape[-1])

    # Flatten memory keys as well as query
    query = query.reshape(-1, query.shape[-1])
    memory_keys = memory_keys.reshape(-1, memory_keys.shape[-1])
    print(f"memory_keys.shape = {memory_keys.shape}")

    # Prepare index for FAISS search
    index.add(memory_keys)

    # FAISS search returns numpy array
    distances, indices = index.search(query, topk)

    # Clip the indices to be within the valid range
    '''
    Here are some common reasons that out of bounds indices can occur in approximate nearest neighbor search:

        The search space is too sparse - There aren't enough near neighbors, so some results spill out of bounds.
        Adding more data points can help.

        Approximation went too far - Algorithms like HNSW provide approximate results for speed. But errors may
        cause some indices to be invalid. Reducing the approximation factor could help.

        Query point is an outlier - If the query itself is very far from the data distribution, even nearest
        neighbors can be out of bounds. Handling outliers in data preprocessing could help.

        Too few nearest neighbors requested - Asking for top-k with low k increases chances of getting invalid
        indices for outliers. Increasing k provides more chances for valid in-bounds points.

        Bugs in implementation - Issues in data preprocessing, such as incorrect bounding boxes, could cause
        unexpected out of bounds indices during search. Checking for bugs could reveal a cause.

        Insufficient clustering - Data indexed without sufficient clustering could have points scattered widely.
        Better clustering before indexing can improve density.

        Incompatible distance metric - A poor distance metric for the data distribution can fail to find true near
        neighbors, again leading to out of bounds indices.

    The core idea is that invalid indices indicate the search failed to find sufficient valid nearby points for
    some queries. Addressing the underlying cause can improve the search density and quality.
    '''
    indices = indices.clip(0, memory_values.size(0)-1)

    # Gather top k values
    memory_values_selected = memory_values[:, indices]

    # Convert distances to torch.Tensor and apply softmax for weights
    weights = torch.from_numpy(numpy.exp(-distances)).to(query.device)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Compute weighted sum of the selected values
    print(f"weights.shape = {weights.shape}")
    print(f"memory_values_selected.shape = {memory_values_selected.shape}")
    aggregated_values = (weights.unsqueeze(-1) * memory_values_selected).sum(dim=-3)
    print(f"aggregated_values.shape = {aggregated_values.shape}")

    # Pass through a linear transformation
    transformed_values = torch.nn.Linear(aggregated_values.size(-1), query.size(-1))(aggregated_values)
    print(f"transformed_values.shape = {transformed_values.shape}")

    # No need to return attn weights/output
    # Just return the retrieved memory values
    return transformed_values


def _gqa_attn(query, key, value, attention_mask=None, scale_attn_weights=False,
              causal_mask_flag=False, dropout=0.0, local_window_size=None, sink_tokens=1,
              fot_mem_keys=None, fot_mem_values=None):
    """Group Query Attention implementation"""


    # Check for potential issues before moving on
    if not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f"Expected query, key, and value to be 4-dimensional, but got shapes "
                         f"{query.shape}, {key.shape}, and {value.shape}.")

    print(f"query_len = {query.shape[2]}, key_len = {key.shape[2]}, value_len = {value.shape[2]}")
    print(f"query'shape = {query.shape}, key's shape = {key.shape}, value's shape = {value.shape}")


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
        print(f"query_grouped.shape = {query_grouped.shape}")

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


    # Incorporate sliding window local attention
    if local_window_size is not None:
        max_seq_len = query.size(-2)
        indices = torch.arange(max_seq_len).to(query.device)
        expanded_indices = indices.unsqueeze(-1).expand(max_seq_len, max_seq_len)
        distance_matrix = torch.abs(expanded_indices - indices.unsqueeze(0))
        print(f"distance_matrix = {distance_matrix}")
        attn_weights.masked_fill_(distance_matrix > local_window_size, float('-inf'))
        print(f"Inside sliding window local attention, attn_weights = {attn_weights}")


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


    # Incorporate FOT memory attention
    if fot_mem_keys is not None and fot_mem_values is not None:
        if n_groups > 1:
            print(f"query_grouped.shape[3] = {query_grouped.shape[3]}")
            fot_mem_values_selected = _fot_mem_attn(query_grouped, fot_mem_keys, fot_mem_values, topk=query_grouped.shape[3])

            # Compute attention weights for FOT memory values
            attn_weights_fot = torch.matmul(query_grouped, fot_mem_keys.transpose(-2, -1))
            attn_weights_fot = attn_weights_fot.sum(dim=1)

        else:
            print(f"query.shape[2] = {query.shape[2]}")
            fot_mem_values_selected = _fot_mem_attn(query, fot_mem_keys, fot_mem_values, topk=query.shape[2])

            # Compute attention weights for FOT memory values
            attn_weights_fot = torch.matmul(query, fot_mem_keys.transpose(-2, -1))

        print(f"fot_mem_keys.transpose(-2, -1)'s shape = {fot_mem_keys.transpose(-2, -1).shape}")
        print(f"fot_mem_values_selected.shape = {fot_mem_values_selected.shape}")

        # Concatenate selected memory values and local context
        print(f"before concat, value.shape = {value.shape}")
        '''
        1. Concatenating along the num_heads dimension (dim 1):
        Pros:
            Allows the attention to spread across both local and global (memory) representations.
            Keeps the sequence dimension aligned between local and global.

        Cons:
            May mix signals across different heads unintentionally.


        2. Concatenating along the sequence dimension (dim 2):
        Pros:
            Keeps local and global values separated by head.
            Allows attention to focus on local vs global.

        Cons:
            Misaligns sequences, need padding potentially.

        The goal of FOT (Focused Transformer) is to allow the model to access both local context
        from the current sequence and global context from long-term memories.

        In most cases, FOT aims for the attention to look collectively across local and global representations
        rather than strongly separating them.
        '''
        fot_mem_values_selected = fot_mem_values_selected.unsqueeze(1)
        fot_mem_values_selected = fot_mem_values_selected.expand(-1, value.shape[1], -1, -1)
        value_with_fot = torch.cat([fot_mem_values_selected, value], dim=1)
        print(f"after concat, value.shape = {value.shape}")

        # Concatenate attention weights
        print(f"Inside FOT attention, attn_weights.shape = {attn_weights.shape}")
        print(f"attn_weights_fot.shape = {attn_weights_fot.shape}")
        print(f"Before concat(), attn_weights.shape = {attn_weights.shape}")
        attn_weights = torch.cat([attn_weights, attn_weights_fot], dim=1)
        print(f"After concat(), attn_weights.shape = {attn_weights.shape}")


    # Softmax normalization to get the attention scores
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)


    # Apply dropout if specified
    if dropout > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout)


    # Compute the output by multiplying the attention scores with the value tensor.
    if fot_mem_keys is not None and fot_mem_values is not None:
        attn_output = torch.matmul(attn_weights, value_with_fot)

    else:
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


    # Define FOT memory length
    fot_mem_len = 6

    # Construct FOT memory keys tensor
    fot_mem_keys = torch.randn(batch_size, fot_mem_len, dim)
    fot_mem_values = torch.randn(batch_size, fot_mem_len, dim)


    # Run the function
    attn_output, attn_weights = \
        _gqa_attn(
                    query, key, value, attention_mask,
                    scale_attn_weights=True, causal_mask_flag=True,
                    dropout=0.0, local_window_size=3, sink_tokens=sink_tokens,
                    fot_mem_keys=fot_mem_keys, fot_mem_values=fot_mem_values
                )


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





