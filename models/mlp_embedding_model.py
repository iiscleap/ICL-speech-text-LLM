import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .embedding_discovery_model import EmbeddingDiscoveryModel

class MLPEmbeddingModel(EmbeddingDiscoveryModel):
    """
    Model that applies a single position-wise MLP transformation to embeddings.
    Uses the same MLP for all tokens, applied position-wise to each embedding vector.
    """
    
    def __init__(self, *args, label_tokens=None, hidden_dim=None, dropout=0.1, **kwargs):
        super().__init__(*args, label_tokens=label_tokens, **kwargs)
        
        # Get embedding dimension
        self.embed_module = self.get_embedding_module()
        self.embed_dim = self.embed_module.weight.shape[1]
        self.hidden_dim = hidden_dim or self.embed_dim
        
        # Create a single position-wise MLP (n×n where n is embedding dimension)
        self.position_wise_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )
        
        # Freeze model parameters, only train the MLP
        self.freeze_base_model()
        logging.info(f"Initialized MLPEmbeddingModel with position-wise MLP (dim: {self.embed_dim}→{self.hidden_dim}→{self.embed_dim})")
    
    def freeze_base_model(self):
        """Freeze all parameters except our MLP"""
        for name, param in self.named_parameters():
            if not name.startswith('position_wise_mlp'):
                param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Model has {trainable_params:,} trainable MLP parameters")

    def forward(self, batch):
        """
        Simple hook that applies MLP to ALL hidden states with proper logging.
        No auxiliary losses, no label token filtering.
        """
        # Find the appropriate layer to hook
        if hasattr(self.llama_model, 'model'):
            if hasattr(self.llama_model.model, 'model'):
                hook_module = self.llama_model.model.model.layers[0]
            else:
                hook_module = self.llama_model.model.layers[0]
        else:
            hook_module = self.llama_model.layers[0]
        
        # Log MLP device/dtype information
        mlp_device = next(self.position_wise_mlp.parameters()).device
        mlp_dtype = next(self.position_wise_mlp.parameters()).dtype
        logging.info(f"[Hook Debug] MLP is on device: {mlp_device}, dtype: {mlp_dtype}")
        
        # Define the hook function
        def pre_hook(module, inputs):
            try:
                if not torch.is_tensor(inputs[0]):
                    return inputs
                
                # Get hidden states
                hidden_states = inputs[0]
                
                # Log hidden states properties
                logging.info(f"[Hook Debug] Hidden states shape: {hidden_states.shape}")
                logging.info(f"[Hook Debug] Hidden states device: {hidden_states.device}, dtype: {hidden_states.dtype}")
                logging.info(f"[Hook Debug] Hidden requires_grad: {hidden_states.requires_grad}")
                
                # Move MLP to match hidden states device
                if mlp_device != hidden_states.device:
                    logging.info(f"[Hook Debug] Moving MLP from {mlp_device} to {hidden_states.device}")
                    self.position_wise_mlp.to(device=hidden_states.device)
                
                # Convert hidden states to float for MLP (if needed)
                orig_dtype = hidden_states.dtype
                if orig_dtype != torch.float32:
                    logging.info(f"[Hook Debug] Converting from {orig_dtype} to float32 for MLP")
                    hidden_float = hidden_states.to(dtype=torch.float32)
                else:
                    hidden_float = hidden_states
                
                # Apply MLP to transform ALL hidden states
                transformed = self.position_wise_mlp(hidden_float)
                logging.info(f"[Hook Debug] Applied MLP transformation")
                
                # Apply residual connection
                result = hidden_float + transformed
                
                # Convert back to original dtype if needed
                if orig_dtype != torch.float32:
                    result = result.to(dtype=orig_dtype)
                    logging.info(f"[Hook Debug] Converted back to {orig_dtype}")
                
                # Return modified inputs
                if len(inputs) > 1:
                    return (result,) + inputs[1:]
                else:
                    return result
                
            except Exception as e:
                logging.error(f"[Hook Debug] Error in hook: {str(e)}", exc_info=True)
                return inputs
        
        # Register the hook
        hook = hook_module.register_forward_pre_hook(pre_hook)
        
        try:
            # Run forward pass - no auxiliary loss, just standard forward
            logging.info(f"[Hook Debug] Starting forward pass with MLP hook")
            outputs = super().forward(batch)
            logging.info(f"[Hook Debug] Completed forward pass")
            
            # Log loss properties
            if hasattr(outputs, "loss"):
                logging.info(f"[Hook Debug] Output loss value: {outputs.loss.item():.6f}")
                logging.info(f"[Hook Debug] Loss requires_grad: {outputs.loss.requires_grad}")
            
            return outputs
        finally:
            # Always remove the hook
            hook.remove()

    def get_transformed_embeddings(self):
        """Get embeddings with transformations applied for label tokens"""
        # Start with a copy of the original embeddings
        transformed_matrix = self.embed_module.weight.clone()
        
        # Get MLP properties
        device = next(self.position_wise_mlp.parameters()).device
        dtype = next(self.position_wise_mlp.parameters()).dtype
        
        # Apply transformation to label tokens
        label_embeds = self.embed_module.weight[self.label_token_ids].to(device=device, dtype=dtype)
        transformed_embeds = self.position_wise_mlp(label_embeds)
        transformed_embeds = label_embeds + transformed_embeds  # residual connection
        
        # Move back to original device and dtype if needed
        if (transformed_embeds.device != transformed_matrix.device or 
            transformed_embeds.dtype != transformed_matrix.dtype):
            transformed_embeds = transformed_embeds.to(
                device=transformed_matrix.device, 
                dtype=transformed_matrix.dtype
            )
        
        # Update in the transformed matrix
        transformed_matrix[self.label_token_ids] = transformed_embeds
        
        return transformed_matrix

    def save_transformed_embeddings(self, path):
        """Save the transformed token embeddings to a file"""
        with torch.no_grad():
            transformed_matrix = self.get_transformed_embeddings()
        
        # Save transformed embeddings with metadata
        torch.save({
            "embeddings": transformed_matrix,
            "vocab_size": transformed_matrix.shape[0],
            "embedding_dim": transformed_matrix.shape[1],
            "label_tokens": self.label_tokens,
            "label_token_ids": self.label_token_ids,
            "mlp_state_dict": self.position_wise_mlp.state_dict()
        }, path)
        logging.info(f"Saved transformed embeddings to {path}")
    
    @classmethod
    def from_custom_salmon(cls, custom_salmon_model, label_tokens=None, hidden_dim=256, dropout=0.1):
        """Create MLPEmbeddingModel from existing CustomSALMONN model"""
        mlp_model = cls(
            device=custom_salmon_model.device,
            low_resource=True,
            label_tokens=label_tokens,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Copy weights from initialized model
        mlp_model.load_state_dict(custom_salmon_model.state_dict(), strict=False)
        
        return mlp_model