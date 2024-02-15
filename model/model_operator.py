import torch
def random_mask_out_dimension(tensor, mask_probability, dim):
    """
    Randomly masks out elements along the specified dimension of the tensor.

    Args:
    - tensor: input tensor of shape (B, C, D, H, W)
    - mask_probability: probability of masking out each element
    - dim: dimension along which to mask out elements (usually set to 2 for dimension D)

    Returns:
    - masked_tensor: tensor with elements randomly masked out along the specified dimension
    """
    # Determine the device of the input tensor
    device = tensor.device
    
    # Generate a random binary mask
    mask = torch.rand(tensor.size(dim), device=device) > mask_probability

    # Expand the mask to the shape of the input tensor along the specified dimension
    mask = mask.view(1, 1, -1, 1, 1).expand(tensor.size())

    # Apply the mask to the input tensor
    masked_tensor = tensor * mask

    return masked_tensor