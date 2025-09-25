#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch  # Import PyTorch tensor library
import torch.nn as nn  # Import PyTorch neural network modules

from typing import Dict  # Import Dict type annotation for type hints

from .layers import LoRALayer  # Import LoRALayer class from the layers module


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Function to mark only LoRA parameters as trainable in the model"""
    for n, p in model.named_parameters():  # Iterate through all named parameters in the model
        if 'lora_' not in n:  # Check if parameter name doesn't contain 'lora_'
            p.requires_grad = False  # Set parameter as non-trainable (frozen)
    if bias == 'none':  # Check if bias parameter handling is set to 'none'
        return  # Exit function early, no bias parameters will be trainable
    elif bias == 'all':  # Check if all bias parameters should be trainable
        for n, p in model.named_parameters():  # Iterate through all named parameters again
            if 'bias' in n:  # Check if parameter name contains 'bias'
                p.requires_grad = True  # Set bias parameter as trainable
    elif bias == 'lora_only':  # Check if only LoRA-related bias parameters should be trainable
        for m in model.modules():  # Iterate through all modules in the model
            # Check if module is a LoRALayer instance and has a non-None bias attribute
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True  # Set LoRA layer bias as trainable
    else:  # Handle invalid bias parameter values
        raise NotImplementedError  # Raise error for unsupported bias options


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """Function to extract only LoRA-related parameters from model's state dictionary"""
    my_state_dict = model.state_dict()  # Get the complete state dictionary from the model
    if bias == 'none':  # Check if no bias parameters should be included
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}  # Return only LoRA parameters
    elif bias == 'all':  # Check if all bias parameters should be included along with LoRA parameters
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}  # Return LoRA and all bias parameters
    elif bias == 'lora_only':  # Check if only LoRA-related bias parameters should be included
        to_return = {}  # Initialize empty dictionary to store selected parameters
        for k in my_state_dict:  # Iterate through all keys in the state dictionary
            if 'lora_' in k:  # Check if key contains 'lora_'
                to_return[k] = my_state_dict[k]  # Add LoRA parameter to return dictionary
                bias_name = k.split('lora_')[0]+'bias'  # Construct corresponding bias parameter name
                if bias_name in my_state_dict:  # Check if corresponding bias parameter exists
                    to_return[bias_name] = my_state_dict[bias_name]  # Add bias parameter to return dictionary
        return to_return  # Return dictionary containing LoRA parameters and their corresponding biases
    else:  # Handle invalid bias parameter values
        raise NotImplementedError  # Raise error for unsupported bias options
