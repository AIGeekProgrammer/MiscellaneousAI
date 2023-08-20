import torch

# Let's assume we process two sentences of length 3 and 4 and a batch size of 2.
# We want to put them into a transformer model with a maximum sequence length of 5.
# The first sentence is: "Był okrutnym siepaczem" and the second one is "Szablą władał jak ręką". Both are Polish.
 
# After tokenization, those the sentences become: 
# ["Był", "okrutnym", "siepaczem", <pad>, <pad>] oraz ["Szablą", "władał", "jak", "ręką", <pad>]

# After embedding (numbers are just for example), we get:
input_tensor = torch.tensor(
                [[0.1, -0.2,  0.3,  0.5], # "Był"
                [0.4,  0.6, -0.1,  0.2],  # "okrutnym"
                [0.7, -0.3,  0.9, -0.2],  # "siepaczem"
                [0.0,  0.0,  0.0,  0.0],  # <pad>
                [0.0,  0.0,  0.0,  0.0],  # <pad>
                [0.1, -0.2,  0.3,  0.5],  # "Szablą"
                [0.4,  0.6, -0.1,  0.2],  # "władał"
                [0.7, -0.3,  0.9, -0.2],  # "jak"
                [-0.5, 0.1,  0.6, -0.3],  # "ręką"
                [0.0,  0.0,  0.0,  0.0]]) # <pad>
# Please note that <pad> token is always 0.

# Create a boolean mask of the same shape as the input tensor, where True indicates a padding token
padding_mask = input_tensor.eq(0.0)
print(padding_mask)

# tensor([[False, False, False, False],
#         [False, False, False, False],
#         [False, False, False, False],
#         [ True,  True,  True,  True],
#         [ True,  True,  True,  True],
#         [False, False, False, False],
#         [False, False, False, False],
#         [False, False, False, False],
#         [False, False, False, False],
#         [ True,  True,  True,  True]])
