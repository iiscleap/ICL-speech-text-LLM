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
        """Convert label tokens to token IDs and maintain token-to-label mapping"""
        token_ids = []
        # Create a mapping from token ID to original label
        self.token_id_to_label = {}
        # Create a mapping from label to all its token IDs
        self.label_to_token_ids = {}
        
        for label in self.label_tokens:
            if isinstance(label, int):
                token_ids.append(label)
                self.token_id_to_label[label] = label
                self.label_to_token_ids[label] = [label]
            else:
                # Token is a string, encode it
                tokenized = self.salmonn.llama_tokenizer.encode(label, add_special_tokens=False)
                if len(tokenized) > 0:
                    # Add ALL token IDs from tokenization
                    for token_id in tokenized:
                        token_ids.append(token_id)
                        self.token_id_to_label[token_id] = label
                    
                    self.label_to_token_ids[label] = tokenized
                    logging.info(f"Label '{label}' tokenized to {len(tokenized)} tokens: {tokenized}")
                    token_texts = [self.salmonn.llama_tokenizer.decode([tid]) for tid in tokenized]
                    logging.info(f"  Decoded tokens: {token_texts}")
                else:
                    logging.warning(f"Could not tokenize '{label}'")
        
        if token_ids:
            logging.info(f"Label token IDs: {token_ids}")
            for token_id in token_ids:
                token_text = self.salmonn.llama_tokenizer.decode([token_id])
                original_label = self.token_id_to_label.get(token_id, "unknown")
                logging.info(f"Token ID {token_id} ('{token_text}') belongs to label '{original_label}'")
        else:
            logging.warning("No valid label tokens provided")
        
        return token_ids
        
    def freeze_all_except_label_embeddings(self):
        """Freeze all parameters except specific label token embeddings"""
        if not self.label_token_ids:
            logging.warning("No label token IDs specified. All parameters will remain frozen.")
            return

        # First, freeze all parameters
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # Get embedding module
        embed_module = self.get_embedding_module()
        
        # Create a mask for the embedding parameters (all zeros)
        embedding_mask = torch.zeros_like(embed_module.weight)
        
        # Set mask to 1 only for the label token rows we want to update
        embedding_mask[self.label_token_ids] = 1.0
        
        # Store the mask for use during training
        self.embedding_mask = embedding_mask
        
        # Only make the embedding weights trainable
        embed_module.weight.requires_grad = True
        
        # Log what we're doing
        trainable_params = len(self.label_token_ids) * embed_module.weight.shape[1]
        logging.info(f"Selectively updating {len(self.label_token_ids)} token embeddings")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
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
        
    def find_nearest_token_embeddings(self, exclude_label_tokens=False, top_k=5, min_similarity=0.0):
        """Find nearest tokens for each individual token in our labels"""
        if not self.label_token_ids:
            return {"error": "No label token IDs available"}
            
        # Get embedding matrix
        embed_module = self.get_embedding_module()
        embed_matrix = embed_module.weight
        
        # Normalize full embedding matrix once for efficiency
        embed_matrix_norm = F.normalize(embed_matrix, p=2, dim=1)
        
        # Prepare results grouped by original label
        results = []
        
        # Process each original label
        for label, token_ids in self.label_to_token_ids.items():
            label_result = {
                "original_label": label,
                "token_results": [],
                "combined_neighbors": []  # We'll fill this with aggregated neighbors
            }
            
            all_neighbors = []  # Collect all neighbors across tokens
            
            # Find neighbors for each token in this label separately
            for token_idx, token_id in enumerate(token_ids):
                # Get token embedding and normalize
                token_embed = embed_matrix[token_id].unsqueeze(0)  # Add batch dimension
                token_embed_norm = F.normalize(token_embed, p=2, dim=1)
                
                # Compute cosine similarity with all vocabulary
                similarity = torch.matmul(token_embed_norm, embed_matrix_norm.transpose(0, 1))[0]
                
                # Exclude original tokens if requested
                if exclude_label_tokens:
                    similarity[token_id] = -1.0
                    
                # Get top-k similar tokens
                topk_values, topk_indices = torch.topk(similarity, k=top_k)
                
                # Get token text
                token_text = self.salmonn.llama_tokenizer.decode([token_id])
                
                # Create neighbors list for this token
                token_neighbors = []
                for j in range(top_k):
                    if topk_values[j] >= min_similarity:
                        neighbor_id = topk_indices[j].item()
                        neighbor_text = self.salmonn.llama_tokenizer.decode([neighbor_id])
                        similarity_score = topk_values[j].item()
                        
                        neighbor_info = {
                            "token_id": neighbor_id,
                            "token_text": neighbor_text,
                            "similarity": similarity_score,
                            "position": token_idx  # Store original position in token sequence
                        }
                        
                        token_neighbors.append(neighbor_info)
                        all_neighbors.append(neighbor_info)  # Add to combined list
                
                # Add this token's results
                label_result["token_results"].append({
                    "token_id": token_id,
                    "token_text": token_text,
                    "neighbors": token_neighbors
                })
            
            # For combined neighbors, first get best neighbor for each position
            position_best_neighbors = {}
            for neighbor in all_neighbors:
                position = neighbor["position"]
                if position not in position_best_neighbors or neighbor["similarity"] > position_best_neighbors[position]["similarity"]:
                    position_best_neighbors[position] = neighbor
            
            # Add neighbors in original token order
            for i in range(len(token_ids)):
                if i in position_best_neighbors:
                    neighbor = position_best_neighbors[i]
                    # Remove position field for final output
                    neighbor_copy = {k: v for k, v in neighbor.items() if k != "position"}
                    label_result["combined_neighbors"].append(neighbor_copy)
            
            results.append(label_result)
        
        # Print results
        for result in results:
            original_label = result["original_label"]
            print(f"\n=== Label: {original_label} ===")
            
            # Print results for each individual token
            for token_result in result["token_results"]:
                token_text = token_result["token_text"]
                print(f"\nToken '{token_text}' nearest neighbors:")
                for i, neighbor in enumerate(token_result["neighbors"][:5]):
                    print(f"  {i+1}. '{neighbor['token_text']}' (similarity: {neighbor['similarity']:.4f})")
            
            # Print combined results
            print(f"\nCombined nearest neighbors for '{original_label}':")
            combined_text = "".join([n["token_text"] for n in result["combined_neighbors"]])
            print(f"  Combined: '{combined_text}'")
            for i, neighbor in enumerate(result["combined_neighbors"]):
                print(f"  {i+1}. '{neighbor['token_text']}' (similarity: {neighbor['similarity']:.4f})")
        
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