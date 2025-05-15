import os
import sys
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .custom_salmon import CustomSALMONN

class EmbeddingDiscoveryModel(CustomSALMONN):
    """
    SALMONN model that only updates specific label token embeddings and finds nearest vocabulary matches.
    """
    
    def __init__(self, *args, label_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store label tokens to track and update
        self.label_tokens = label_tokens or []
        self.label_token_ids = None
        if self.label_tokens:
            self.label_token_ids = self._get_label_token_ids()
            
        # Set flag for embedding discovery mode
        self.embedding_discovery_mode = True
        self.freeze_all_except_label_embeddings()
        logging.info("Initialized EmbeddingDiscoveryModel with frozen parameters except label embeddings")
    
    def _get_label_token_ids(self):
        """Convert label tokens to token IDs"""
        token_ids = []
        for token in self.label_tokens:
            if isinstance(token, int):
                token_ids.append(token)
            else:
                # Token is a string, encode it
                tokenized = self.salmonn.llama_tokenizer.encode(token, add_special_tokens=False)
                if len(tokenized) > 0:
                    token_ids.append(tokenized[0])
                else:
                    logging.warning(f"Could not tokenize '{token}'")
        
        if token_ids:
            logging.info(f"Label token IDs: {token_ids}")
            for token_id in token_ids:
                token_text = self.salmonn.llama_tokenizer.decode([token_id])
                logging.info(f"Token ID {token_id} corresponds to '{token_text}'")
        else:
            logging.warning("No valid label tokens provided")
        
        return token_ids
        
    def freeze_all_except_label_embeddings(self):
        """Freeze all parameters except specific label token embeddings"""
        # Count parameters before freezing
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Freeze all parameters first
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        if not self.label_token_ids:
            logging.warning("No label token IDs specified. All parameters will remain frozen.")
            return
            
        # Get embedding module and matrix
        embed_module = self.get_embedding_module()
        
        # Create selective gradient hook for embedding matrix
        def select_grad_hook(grad):
            # Create a mask of zeros
            mask = torch.zeros_like(grad)
            # Set mask to 1 for label token indices
            mask[self.label_token_ids] = 1.0
            # Zero out gradients for non-label tokens
            return grad * mask
        
        # Register hook and make embedding matrix trainable
        embed_module.weight.requires_grad = True
        embed_module.weight.register_hook(select_grad_hook)
            
        # Count parameters after freezing
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        effective_params = len(self.label_token_ids) * embed_module.weight.shape[1]
        
        logging.info(f"Froze all parameters except embeddings for label tokens: {self.label_tokens}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Effective trainable parameters: {effective_params:,} (only label token embeddings will be updated)")
        
    def get_embedding_module(self):
        """Get the token embedding module based on model structure"""
        if hasattr(self.salmonn, 'llama_model'):
            # Check if using LoRA
            if hasattr(self.salmonn.llama_model, 'base_model'):
                model = self.salmonn.llama_model.base_model.model
            else:
                model = self.salmonn.llama_model.model
                
            # Access embeddings based on model structure
            if hasattr(model, 'model'):
                embed_module = model.model.embed_tokens
            else:
                embed_module = model.embed_tokens
                
            return embed_module
        else:
            raise ValueError("Could not locate embedding module in model structure")
        
    def find_nearest_token_embeddings(self, exclude_label_tokens=False, top_k=1, min_similarity=0.0):
        """
        Find the nearest vocabulary token embeddings to our trained label embeddings
        
        Args:
            exclude_label_tokens: Whether to exclude the original label tokens from results
            top_k: Number of nearest neighbors to return
            min_similarity: Minimum similarity threshold for inclusion
            
        Returns:
            List of results with nearest tokens for each label
        """
        if not self.label_token_ids:
            return {"error": "No label token IDs available"}
            
        # Get the embedding matrix
        embed_module = self.get_embedding_module()
        embed_matrix = embed_module.weight
        
        # Get label embeddings
        label_embeddings = embed_matrix[self.label_token_ids]
        
        # Normalize embeddings for cosine similarity
        embed_matrix_norm = F.normalize(embed_matrix, p=2, dim=1)
        label_embeds_norm = F.normalize(label_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(label_embeds_norm, embed_matrix_norm.transpose(0, 1))
        
        # Create exclusion mask for the label tokens themselves if requested
        if exclude_label_tokens:
            for i, token_id in enumerate(self.label_token_ids):
                similarity[i, token_id] = -1.0  # Exclude from top-k
        
        # Get top-k similar tokens
        topk_values, topk_indices = torch.topk(similarity, k=top_k, dim=1)
        
        # Prepare results
        results = []
        for i, token_id in enumerate(self.label_token_ids):
            token_text = self.salmonn.llama_tokenizer.decode([token_id])
            neighbors = []
            
            for j in range(top_k):
                if topk_values[i, j] >= min_similarity:
                    neighbor_id = topk_indices[i, j].item()
                    neighbor_text = self.salmonn.llama_tokenizer.decode([neighbor_id])
                    similarity_score = topk_values[i, j].item()
                    
                    neighbors.append({
                        "token_id": neighbor_id,
                        "token_text": neighbor_text,
                        "similarity": similarity_score
                    })
                
            results.append({
                "original_label": self.label_tokens[i] if i < len(self.label_tokens) else "unknown",
                "target_token": token_text,
                "target_id": token_id,
                "neighbors": neighbors
            })
            
        return results
    
    def save_token_embeddings(self, path):
        """Save the token embeddings to a file"""
        embed_module = self.get_embedding_module()
        embed_matrix = embed_module.weight
        
        # Save all embeddings plus metadata about which were updated
        torch.save({
            "embeddings": embed_matrix,
            "vocab_size": embed_matrix.shape[0],
            "embedding_dim": embed_matrix.shape[1],
            "label_tokens": self.label_tokens,
            "label_token_ids": self.label_token_ids
        }, path)
        logging.info(f"Saved token embeddings to {path}")
    
    def print_token_info(self, tokens):
        """Print tokenization information for debugging"""
        tokenizer = self.salmonn.llama_tokenizer
        
        for token in tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            token_texts = [tokenizer.decode([tid]) for tid in token_ids]
            
            logging.info(f"Token '{token}':")
            logging.info(f"  Token IDs: {token_ids}")
            logging.info(f"  Decoded tokens: {token_texts}")
        
    @classmethod
    def from_custom_salmon(cls, custom_salmon_model, label_tokens=None):
        """Create an EmbeddingDiscoveryModel from an existing CustomSALMONN model"""
        discovery_model = cls(
            device=custom_salmon_model.device,
            low_resource=True,
            label_tokens=label_tokens
        )
        
        # Copy weights from initialized model
        discovery_model.load_state_dict(custom_salmon_model.state_dict())
        
        return discovery_model