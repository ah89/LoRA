#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch  # Import PyTorch tensor library
import torch.nn as nn  # Import PyTorch neural network modules
import torch.nn.functional as F  # Import PyTorch functional interface for operations

import math  # Import Python math module for mathematical functions
from typing import Optional, List  # Import type annotations for optional and list types

class LoRALayer():
    """Base class for LoRA (Low-Rank Adaptation) layers"""
    def __init__(
        self, 
        r: int,  # Rank of the adaptation - lower rank means fewer parameters
        lora_alpha: int,  # Scaling parameter for LoRA adaptation
        lora_dropout: float,  # Dropout rate applied to LoRA layers
        merge_weights: bool,  # Whether to merge LoRA weights with original weights during inference
    ):
        self.r = r  # Store the rank parameter
        self.lora_alpha = lora_alpha  # Store the alpha scaling parameter
        # Optional dropout
        if lora_dropout > 0.:  # Check if dropout rate is greater than zero
            self.lora_dropout = nn.Dropout(p=lora_dropout)  # Create dropout layer with specified rate
        else:  # If no dropout is needed
            self.lora_dropout = lambda x: x  # Create identity function (no dropout)
        # Mark the weight as unmerged
        self.merged = False  # Initialize merged status as False (weights are separate)
        self.merge_weights = merge_weights  # Store the merge_weights flag


class Embedding(nn.Embedding, LoRALayer):
    """LoRA implementation for embedding layers - inherits from both nn.Embedding and LoRALayer"""
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,  # Size of the dictionary of embeddings
        embedding_dim: int,  # The size of each embedding vector
        r: int = 0,  # Rank of LoRA adaptation (0 means no LoRA)
        lora_alpha: int = 1,  # LoRA scaling parameter
        merge_weights: bool = True,  # Whether to merge weights during inference
        **kwargs  # Additional arguments passed to nn.Embedding
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)  # Initialize parent nn.Embedding
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,  # Initialize LoRALayer parent
                           merge_weights=merge_weights)  # Pass parameters to LoRALayer
        # Actual trainable parameters
        if r > 0:  # Only create LoRA parameters if rank is greater than 0
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))  # Create LoRA matrix A with shape (r, vocab_size)
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))  # Create LoRA matrix B with shape (embed_dim, r)
            self.scaling = self.lora_alpha / self.r  # Calculate scaling factor for LoRA adaptation
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False  # Freeze the original embedding weights
        self.reset_parameters()  # Initialize all parameters

    def reset_parameters(self):
        """Reset/initialize parameters for the embedding layer"""
        nn.Embedding.reset_parameters(self)  # Call parent class parameter reset
        if hasattr(self, 'lora_A'):  # Check if LoRA parameters exist
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)  # Initialize LoRA matrix A with zeros
            nn.init.normal_(self.lora_B)  # Initialize LoRA matrix B with normal distribution

    def train(self, mode: bool = True):
        """Set the module in training or evaluation mode and handle weight merging"""
        nn.Embedding.train(self, mode)  # Set parent embedding layer to training/eval mode
        if mode:  # If switching to training mode
            if self.merge_weights and self.merged:  # Check if weights are currently merged and should be separated
                # Make sure that the weights are not merged
                if self.r > 0:  # Only if LoRA is enabled (rank > 0)
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling  # Subtract LoRA adaptation from merged weights
                self.merged = False  # Mark weights as unmerged
        else:  # If switching to evaluation mode
            if self.merge_weights and not self.merged:  # Check if weights should be merged and aren't already
                # Merge the weights and mark it
                if self.r > 0:  # Only if LoRA is enabled (rank > 0)
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling  # Add LoRA adaptation to original weights
                self.merged = True  # Mark weights as merged
        
    def forward(self, x: torch.Tensor):
        """Forward pass through the LoRA embedding layer"""
        if self.r > 0 and not self.merged:  # If LoRA is enabled and weights are not merged
            result = nn.Embedding.forward(self, x)  # Get embedding from original weight matrix
            after_A = F.embedding(  # Apply LoRA matrix A using functional embedding
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,  # Input tensor and transposed lora_A matrix
                self.norm_type, self.scale_grad_by_freq, self.sparse  # Additional embedding parameters
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling  # Add LoRA adaptation (A*B) scaled appropriately
            return result  # Return the combined result
        else:  # If LoRA is disabled or weights are merged
            return nn.Embedding.forward(self, x)  # Use standard embedding forward pass
            

class Linear(nn.Linear, LoRALayer):
    """LoRA implementation for linear/dense layers - inherits from both nn.Linear and LoRALayer"""
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        r: int = 0,  # Rank of LoRA adaptation (0 means no LoRA)
        lora_alpha: int = 1,  # LoRA scaling parameter
        lora_dropout: float = 0.,  # Dropout rate for LoRA layers
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,  # Whether to merge weights during inference
        **kwargs  # Additional arguments passed to nn.Linear
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)  # Initialize parent nn.Linear
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,  # Initialize LoRALayer parent
                           merge_weights=merge_weights)  # Pass parameters to LoRALayer

        self.fan_in_fan_out = fan_in_fan_out  # Store the fan_in_fan_out flag
        # Actual trainable parameters
        if r > 0:  # Only create LoRA parameters if rank is greater than 0
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))  # Create LoRA matrix A with shape (r, in_features)
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))  # Create LoRA matrix B with shape (out_features, r)
            self.scaling = self.lora_alpha / self.r  # Calculate scaling factor for LoRA adaptation
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False  # Freeze the original linear layer weights
        self.reset_parameters()  # Initialize all parameters
        if fan_in_fan_out:  # If weights should be transposed (for certain layer types)
            self.weight.data = self.weight.data.transpose(0, 1)  # Transpose the weight matrix

    def reset_parameters(self):
        """Reset/initialize parameters for the linear layer"""
        nn.Linear.reset_parameters(self)  # Call parent class parameter reset
        if hasattr(self, 'lora_A'):  # Check if LoRA parameters exist
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Initialize LoRA matrix A with Kaiming uniform distribution
            nn.init.zeros_(self.lora_B)  # Initialize LoRA matrix B with zeros

    def train(self, mode: bool = True):
        """Set the module in training or evaluation mode and handle weight merging"""
        def T(w):  # Helper function for transposing weights if needed
            return w.transpose(0, 1) if self.fan_in_fan_out else w  # Transpose if fan_in_fan_out is True, otherwise return as-is
        nn.Linear.train(self, mode)  # Set parent linear layer to training/eval mode
        if mode:  # If switching to training mode
            if self.merge_weights and self.merged:  # Check if weights are currently merged and should be separated
                # Make sure that the weights are not merged
                if self.r > 0:  # Only if LoRA is enabled (rank > 0)
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling  # Subtract LoRA adaptation from merged weights
                self.merged = False  # Mark weights as unmerged
        else:  # If switching to evaluation mode
            if self.merge_weights and not self.merged:  # Check if weights should be merged and aren't already
                # Merge the weights and mark it
                if self.r > 0:  # Only if LoRA is enabled (rank > 0)
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling  # Add LoRA adaptation to original weights
                self.merged = True  # Mark weights as merged       

    def forward(self, x: torch.Tensor):
        """Forward pass through the LoRA linear layer"""
        def T(w):  # Helper function for transposing weights if needed
            return w.transpose(0, 1) if self.fan_in_fan_out else w  # Transpose if fan_in_fan_out is True, otherwise return as-is
        if self.r > 0 and not self.merged:  # If LoRA is enabled and weights are not merged
            result = F.linear(x, T(self.weight), bias=self.bias)  # Apply linear transformation using original weights
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling  # Add LoRA adaptation: dropout(x) @ A^T @ B^T * scaling
            return result  # Return the combined result
        else:  # If LoRA is disabled or weights are merged
            return F.linear(x, T(self.weight), bias=self.bias)  # Use standard linear transformation


class MergedLinear(nn.Linear, LoRALayer):
    """LoRA implementation for merged linear layers that can selectively apply LoRA to different output heads"""
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        r: int = 0,  # Rank of LoRA adaptation (0 means no LoRA)
        lora_alpha: int = 1,  # LoRA scaling parameter
        lora_dropout: float = 0.,  # Dropout rate for LoRA layers
        enable_lora: List[bool] = [False],  # List indicating which output heads should have LoRA applied
        fan_in_fan_out: bool = False,  # Set this to True if the layer stores weight like (fan_in, fan_out)
        merge_weights: bool = True,  # Whether to merge weights during inference
        **kwargs  # Additional arguments passed to nn.Linear
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)  # Initialize parent nn.Linear
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,  # Initialize LoRALayer parent
                           merge_weights=merge_weights)  # Pass parameters to LoRALayer
        # Ensure output features can be evenly divided among heads
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora  # Store which heads should have LoRA applied
        self.fan_in_fan_out = fan_in_fan_out  # Store the fan_in_fan_out flag
        # Actual trainable parameters
        if r > 0 and any(enable_lora):  # Only create LoRA parameters if rank > 0 and at least one head has LoRA enabled
            self.lora_A = nn.Parameter(  # Create LoRA matrix A parameter
                self.weight.new_zeros((r * sum(enable_lora), in_features)))  # Shape: (rank * num_lora_heads, in_features)
            self.lora_B = nn.Parameter(  # Create LoRA matrix B parameter
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))  # Shape optimized for grouped convolution
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r  # Calculate scaling factor for LoRA adaptation
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False  # Freeze the original linear layer weights
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(  # Create boolean mask for LoRA-enabled outputs
                (out_features, ), dtype=torch.bool  # Initialize with zeros and boolean type
            ).view(len(enable_lora), -1)  # Reshape to match the number of heads
            self.lora_ind[enable_lora, :] = True  # Set True for heads that have LoRA enabled
            self.lora_ind = self.lora_ind.view(-1)  # Flatten back to 1D for indexing
        self.reset_parameters()  # Initialize all parameters
        if fan_in_fan_out:  # If weights should be transposed
            self.weight.data = self.weight.data.transpose(0, 1)  # Transpose the weight matrix

    def reset_parameters(self):
        """Reset/initialize parameters for the merged linear layer"""
        nn.Linear.reset_parameters(self)  # Call parent class parameter reset
        if hasattr(self, 'lora_A'):  # Check if LoRA parameters exist
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Initialize LoRA matrix A with Kaiming uniform distribution
            nn.init.zeros_(self.lora_B)  # Initialize LoRA matrix B with zeros

    def zero_pad(self, x):
        """Apply zero padding to align LoRA outputs with the full output tensor"""
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))  # Create zero tensor with full output dimensions
        result[self.lora_ind] = x  # Fill in the LoRA-enabled positions with actual values
        return result  # Return the zero-padded result

    def merge_AB(self):
        """Merge LoRA matrices A and B using grouped 1D convolution"""
        def T(w):  # Helper function for transposing weights if needed
            return w.transpose(0, 1) if self.fan_in_fan_out else w  # Transpose if fan_in_fan_out is True, otherwise return as-is
        delta_w = F.conv1d(  # Use 1D convolution to efficiently compute A @ B for multiple heads
            self.lora_A.unsqueeze(0),  # Add batch dimension to lora_A: (1, r*num_heads, in_features)
            self.lora_B.unsqueeze(-1),  # Add kernel dimension to lora_B: (out_per_head*num_heads, r, 1)
            groups=sum(self.enable_lora)  # Use grouped convolution with one group per LoRA-enabled head
        ).squeeze(0)  # Remove batch dimension: (out_per_head*num_heads, in_features)
        return T(self.zero_pad(delta_w))  # Apply zero padding and optional transpose

    def train(self, mode: bool = True):
        """Set the module in training or evaluation mode and handle weight merging"""
        def T(w):  # Helper function for transposing weights if needed
            return w.transpose(0, 1) if self.fan_in_fan_out else w  # Transpose if fan_in_fan_out is True, otherwise return as-is
        nn.Linear.train(self, mode)  # Set parent linear layer to training/eval mode
        if mode:  # If switching to training mode
            if self.merge_weights and self.merged:  # Check if weights are currently merged and should be separated
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):  # Only if LoRA is enabled and at least one head uses LoRA
                    self.weight.data -= self.merge_AB() * self.scaling  # Subtract merged LoRA weights from original weights
                self.merged = False  # Mark weights as unmerged
        else:  # If switching to evaluation mode
            if self.merge_weights and not self.merged:  # Check if weights should be merged and aren't already
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):  # Only if LoRA is enabled and at least one head uses LoRA
                    self.weight.data += self.merge_AB() * self.scaling  # Add merged LoRA weights to original weights
                self.merged = True  # Mark weights as merged        

    def forward(self, x: torch.Tensor):
        """Forward pass through the merged LoRA linear layer"""
        def T(w):  # Helper function for transposing weights if needed
            return w.transpose(0, 1) if self.fan_in_fan_out else w  # Transpose if fan_in_fan_out is True, otherwise return as-is
        if self.merged:  # If weights are merged (typically during inference)
            return F.linear(x, T(self.weight), bias=self.bias)  # Use standard linear transformation with merged weights
        else:  # If weights are separate (typically during training)
            result = F.linear(x, T(self.weight), bias=self.bias)  # Apply linear transformation using original weights
            if self.r > 0:  # Only add LoRA adaptation if rank > 0
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling  # Add LoRA adaptation: dropout(x) @ (merged_AB)^T * scaling
            return result  # Return the combined result

class ConvLoRA(nn.Module, LoRALayer):
    """Base class for convolutional LoRA layers"""
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()  # Initialize nn.Module parent class
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)  # Create the base convolution layer
        for name, param in self.conv.named_parameters():  # Iterate through all parameters of the conv layer
            self.register_parameter(name, param)  # Register each parameter with this module
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)  # Initialize LoRALayer parent
        assert isinstance(kernel_size, int)  # Ensure kernel_size is an integer (not tuple)
        # Actual trainable parameters
        if r > 0:  # Only create LoRA parameters if rank is greater than 0
            self.lora_A = nn.Parameter(  # Create LoRA matrix A parameter
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))  # Shape accommodates kernel dimensions
            )
            self.lora_B = nn.Parameter(  # Create LoRA matrix B parameter
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))  # Shape considers conv groups
            )
            self.scaling = self.lora_alpha / self.r  # Calculate scaling factor for LoRA adaptation
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False  # Freeze the original convolution weights
        self.reset_parameters()  # Initialize all parameters
        self.merged = False  # Initialize merged status as False

    def reset_parameters(self):
        """Reset/initialize parameters for the convolutional LoRA layer"""
        self.conv.reset_parameters()  # Reset the base convolution layer parameters
        if hasattr(self, 'lora_A'):  # Check if LoRA parameters exist
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Initialize LoRA matrix A with Kaiming uniform distribution
            nn.init.zeros_(self.lora_B)  # Initialize LoRA matrix B with zeros

    def train(self, mode=True):
        """Set the module in training or evaluation mode and handle weight merging"""
        super(ConvLoRA, self).train(mode)  # Set parent module to training/eval mode
        if mode:  # If switching to training mode
            if self.merge_weights and self.merged:  # Check if weights are currently merged and should be separated
                if self.r > 0:  # Only if LoRA is enabled (rank > 0)
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling  # Subtract LoRA adaptation reshaped to conv weight dimensions
                self.merged = False  # Mark weights as unmerged
        else:  # If switching to evaluation mode
            if self.merge_weights and not self.merged:  # Check if weights should be merged and aren't already
                if self.r > 0:  # Only if LoRA is enabled (rank > 0)
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling  # Add LoRA adaptation reshaped to conv weight dimensions
                self.merged = True  # Mark weights as merged

    def forward(self, x):
        """Forward pass through the convolutional LoRA layer"""
        if self.r > 0 and not self.merged:  # If LoRA is enabled and weights are not merged
            return self.conv._conv_forward(  # Use the internal convolution forward method
                x,  # Input tensor
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,  # Original weights + LoRA adaptation reshaped and scaled
                self.conv.bias  # Bias term
            )
        return self.conv(x)  # Use standard convolution if LoRA disabled or weights merged

class Conv2d(ConvLoRA):
    """LoRA implementation for 2D convolutional layers"""
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)  # Initialize with nn.Conv2d as the base convolution module

class Conv1d(ConvLoRA):
    """LoRA implementation for 1D convolutional layers"""
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)  # Initialize with nn.Conv1d as the base convolution module

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    """LoRA implementation for 3D convolutional layers"""
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)  # Initialize with nn.Conv3d as the base convolution module
