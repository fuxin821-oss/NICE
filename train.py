import os, re, pdb
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline
from src.utils import seed_everything
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

alpha = 0.4


def get_token_id(prompt, tokenizer=None, return_ids_only=True):
    token_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids

def get_unconditional_target(pipeline, target_concepts, device):
    """获取用于无条件引导的目标概念方向"""
    target_embs = []
    for concept in target_concepts:
        inputs = get_token_id(concept, pipeline.tokenizer, return_ids_only=False)
        emb = pipeline.text_encoder(inputs.input_ids.to(device)).last_hidden_state[0]
        # 取最后一个subject token
        last_token_idx = inputs.attention_mask[0].sum().item() - 2
        target_embs.append(emb[[last_token_idx], :])
    
    return torch.cat(target_embs, dim=0).mean(dim=0, keepdim=True)  # [1, emb_dim]

def generate_perturbed_embs(ret_embs, P, erase_weight, num_per_sample, mini_batch=8):
    ret_embs = ret_embs.squeeze(1)
    out_embs, norm_list = [], []
    for i in range(0, ret_embs.size(0), mini_batch):
        mini_ret_embs = ret_embs[i:i + mini_batch]
        for _ in range(num_per_sample):
            noise = torch.randn_like(mini_ret_embs)
            perturbed_embs = mini_ret_embs + noise @ P
            out_embs.append(perturbed_embs)
            norm_list.append(torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1))
    out_embs = torch.cat(out_embs, dim=0)
    norm_list = torch.cat(norm_list, dim=0)
    return out_embs[norm_list > norm_list.mean()].unsqueeze(1) # shape: [Num, 1, 768]


def extract_style_from_target(target_embs, anchor_embs, num_style_components=10):   # 3
    """
    Extract style components from target embeddings
    
    Args:
        target_embs: Target concept embeddings
        anchor_embs: Anchor concept embeddings
        num_style_components: Number of style components to extract
        
    Returns:
        Style components extracted from target embeddings
    """
    # Flatten embeddings
    target_flat = target_embs.reshape(-1, target_embs.shape[-1])
    
    # Simple approach: use PCA-like decomposition to extract style components
    # Compute the covariance matrix
    target_centered = target_flat - target_flat.mean(dim=0, keepdim=True)
    cov = target_centered.T @ target_centered
    
    # Get principal components through SVD
    U, S, V = torch.svd(cov)
    
    # Select the top principal components as style components
    # These are directions of maximum variance in the data
    style_components = U[:, :num_style_components]
    
    
    # Ensure style_components has shape [num_components, emb_dim] for concatenation
    return style_components.T  # Returns shape [num_components, emb_dim]
class MultiLevelSemanticPreserver:
    """在多层级上进行语义保持"""
    
    def __init__(self, pipeline, device, semantic_layers=None):
        self.pipeline = pipeline
        self.device = device
        self.original_unet_state = {
            name: param.data.clone().detach() 
            for name, param in pipeline.unet.named_parameters()
        }
        
        self.semantic_layers = semantic_layers or [
            'down_blocks.0.attentions.0.transformer_blocks.0.attn2',  
            'down_blocks.2.attentions.0.transformer_blocks.0.attn2',    
            'mid_block.attentions.0.transformer_blocks.0.attn2',      
            'up_blocks.1.attentions.0.transformer_blocks.0.attn2',  
        ]
        self.feature_maps = {}
        self.hooks = []
        
    def register_hooks(self):
        hooked_layers = 0
        for name, module in self.pipeline.unet.named_modules():
            if name in self.semantic_layers and hasattr(module, 'register_forward_hook'):
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                hooked_layers += 1
              
        
       
        if hooked_layers == 0:
            print("⚠ 警告：没有找到任何可注册的层！")
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.feature_maps[name] = output.detach().clone()
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_multi_level_similarity(self, original_features, edited_features):
        similarity_loss = 0
        weights = {
            'down_blocks.0.attentions.0.transformer_blocks.0.attn2': 0.1,
            'down_blocks.2.attentions.0.transformer_blocks.0.attn2': 0.2,
            'mid_block.attentions.0.transformer_blocks.0.attn2': 0.4,
            'up_blocks.1.attentions.0.transformer_blocks.0.attn2': 0.3
        }
        matched_layers = 0
        
         
        for layer_name, orig_feat in original_features.items():
            if layer_name in edited_features and layer_name in weights:
                edited_feat = edited_features[layer_name]
                matched_layers += 1
                
                layer_weight = weights[layer_name]
                
                if orig_feat.numel() > 0 and edited_feat.numel() > 0:
                    orig_flat = orig_feat.flatten()
                    edited_flat = edited_feat.flatten()
                    
                    min_len = min(len(orig_flat), len(edited_flat))
                    if min_len == 0:
                        continue
                        
                    orig_flat = orig_flat[:min_len]
                    edited_flat = edited_flat[:min_len]
                    
                    cosine_sim = F.cosine_similarity(orig_flat.unsqueeze(0), edited_flat.unsqueeze(0), dim=1)
                    mse_loss = F.mse_loss(orig_flat, edited_flat)
                    
                    layer_loss = layer_weight * (1 - cosine_sim + mse_loss)
                    similarity_loss += layer_loss
                   
        if matched_layers == 0:
            return torch.tensor(0.0, device=self.device)
        
        avg_loss = similarity_loss / matched_layers
        return avg_loss
    
    def get_original_features(self, prompt, latent, timestep):
        current_state = {name: param.data.clone() for name, param in self.pipeline.unet.named_parameters()}
       
        for name, param in self.pipeline.unet.named_parameters():
            if name in self.original_unet_state:
                param.data.copy_(self.original_unet_state[name])
        
        self.feature_maps.clear()
        with torch.no_grad():
            _ = self.pipeline.unet(
                latent, timestep, 
                encoder_hidden_states=self.pipeline.text_encoder(
                    get_token_id(prompt, self.pipeline.tokenizer, return_ids_only=False).input_ids.to(self.device)
                ).last_hidden_state
            )
        original_features = {k: v.clone() for k, v in self.feature_maps.items()}
        
        for name, param in self.pipeline.unet.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])
        
        return original_features
    
    def check_model_difference(self):
        total_diff = 0
        changed_params = 0
        
        for name, param in self.pipeline.unet.named_parameters():
            if name in self.original_unet_state:
                diff = torch.norm(param.data - self.original_unet_state[name])
                total_diff += diff.item()
                if diff.item() > 1e-6:  
                    changed_params += 1
      
        return total_diff
    
    def get_semantic_preservation_loss(self, prompt, latent, timestep):
        original_features = self.get_original_features(prompt, latent, timestep)
        self.feature_maps.clear()
        with torch.no_grad():
            _ = self.pipeline.unet(
                latent, timestep,
                encoder_hidden_states=self.pipeline.text_encoder(
                    get_token_id(prompt, self.pipeline.tokenizer, return_ids_only=False).input_ids.to(self.device)
                ).last_hidden_state
            )
        current_features = {k: v.clone() for k, v in self.feature_maps.items()}
        loss = self.compute_multi_level_similarity(original_features, current_features)
        
        return loss
def adaptive_layer_weighting(layer_name, total_layers):
    if 'down' in layer_name:
        return 0.6
    elif 'mid' in layer_name:
        return 0.3
    elif 'up' in layer_name:
        return 0.2
    else:
        return 0.1
    
@torch.no_grad()
def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, baseline=None, chunk_size=128, emb_size=768, device="cuda"):
    enable_semantic = args.semantic_lambda > 0 and retain_texts and retain_texts[0].strip()
    
    if enable_semantic:
        semantic_preserver = MultiLevelSemanticPreserver(pipeline, device)
        semantic_preserver.register_hooks()
        initial_diff = semantic_preserver.check_model_difference()
        test_latent = torch.randn(1, 4, 64, 64, device=device)
        test_timestep = torch.tensor([500], device=device)
        test_prompt = retain_texts[0] if retain_texts[0].strip() else "a photo"
        
        test_loss = semantic_preserver.get_semantic_preservation_loss(test_prompt, test_latent, test_timestep)
      
        if test_loss.item() == 0:
            print("⚠ 警告：初始语义损失为0，可能存在问题！")
    else:
        semantic_preserver = None
       

    if args.ace_enable:
        unconditional_target = get_unconditional_target(pipeline, target_concepts, device)
       
    
    I = torch.eye(emb_size, device=device)
    if args.params == 'KV':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}
    elif args.params == 'V':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_v' in k}
    elif args.params == 'K':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k}
 

    if baseline == 'SPEED':
        null_inputs = get_token_id('', pipeline.tokenizer, return_ids_only=False)
        null_hidden = pipeline.text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0]
        cluster_ids, cluster_centers = kmeans(X=null_hidden[1:], num_clusters=3, distance='euclidean', device='cuda')
        
        # Get target and anchor embeddings for style extraction
        all_target_embs, all_anchor_embs = [], []
        for i in range(len(target_concepts)):
            target_inputs = get_token_id(target_concepts[i], pipeline.tokenizer, return_ids_only=False)
            target_embs = pipeline.text_encoder(target_inputs.input_ids.to(device)).last_hidden_state[0]
            anchor_inputs = get_token_id(anchor_concepts[i], pipeline.tokenizer, return_ids_only=False)
            anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0]
            
            if target_concepts == ['nudity']:
                all_target_embs.append(target_embs[1:, :])  
                all_anchor_embs.append(anchor_embs[1:, :])  
            else:
                all_target_embs.append(target_embs[[(target_inputs.attention_mask[0].sum().item() - 2)], :])  
                all_anchor_embs.append(anchor_embs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]) 
        
        all_target_embs = torch.cat(all_target_embs, dim=0)
        all_anchor_embs = torch.cat(all_anchor_embs, dim=0)
        
        # Extract style components from target embeddings
        if hasattr(args, 'style_components') and args.style_components > 0:
            style_components = extract_style_from_target(all_target_embs, all_anchor_embs, num_style_components=args.style_components)
            K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device), style_components], dim=0).T
        else:
            K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T
        
        I2 = torch.eye(len(K2.T), device=device)
    else:
        raise ValueError("Invalid baseline")

    # region [Target and Anchor]
    sum_anchor_target, sum_target_target = [], []
    for i in range(0, len(target_concepts)):
        target_inputs = get_token_id(target_concepts[i], pipeline.tokenizer, return_ids_only=False)
        target_embs = pipeline.text_encoder(target_inputs.input_ids.to(device)).last_hidden_state[0]
        anchor_inputs = get_token_id(anchor_concepts[i], pipeline.tokenizer, return_ids_only=False)
        anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0]
        if target_concepts == ['nudity']:
            target_embs = target_embs[1:, :]  
            anchor_embs = anchor_embs[1:, :]  
        else:
            target_embs = target_embs[[(target_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
            anchor_embs = anchor_embs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
            anchor_embs = anchor_embs * (1 - alpha * target_embs @ anchor_embs.T / (anchor_embs.norm()**2)) + target_embs * alpha
        sum_target_target.append(target_embs.T @ target_embs)      
        sum_anchor_target.append(anchor_embs.T @ target_embs)      
    sum_target_target, sum_anchor_target = torch.stack(sum_target_target).mean(0), torch.stack(sum_anchor_target).mean(0)
    
 
    retain_texts = [text for text in retain_texts if not any(re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()) for concept in target_concepts)]
    
        
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts))
    last_ret_embs = []
    for j in range(0, len(retain_texts), chunk_size):
        ret_inputs = get_token_id(retain_texts[j:j + chunk_size], pipeline.tokenizer, return_ids_only=False)
        ret_embs_batch = pipeline.text_encoder(ret_inputs.input_ids.to(device)).last_hidden_state
        if retain_texts == ['']: 
            last_ret_embs.append(ret_embs_batch[:, 1:, :].permute(1, 0, 2))
        else:
            last_subject_indices = ret_inputs.attention_mask.sum(1) - 2
            last_ret_embs.append(ret_embs_batch[torch.arange(ret_embs_batch.size(0)), last_subject_indices].unsqueeze(1))
    last_ret_embs = torch.cat(last_ret_embs)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]  # shuffle
    # endregion
    test_latents = [torch.randn(1, 4, 64, 64, device=device) for _ in range(3)]
    test_timesteps = [torch.tensor([t], device=device) for t in [100, 500, 900]]

    semantic_applied_count = 0

    for layer_idx, (layer_name, layer_weight) in enumerate(tqdm(edit_dict.items(), desc="Model Editing")):
        layer_weight_factor = adaptive_layer_weighting(layer_name, len(edit_dict))
   
        base_erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()
        
        if args.ace_enable:
            unconditional_erase_component = layer_weight @ (-unconditional_target.T @ unconditional_target) @ (I + sum_target_target).inverse()
            erase_weight = base_erase_weight + args.ace_lambda * unconditional_erase_component
        else:
            erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse() # Δerase = W(C*·C1^T - C1·C1^T)(I + C1·C1^T)^(-1)
        (U0, S0, V0) = torch.svd(layer_weight)   # SVD(W)
        P0_min = V0[:, -1:] @ V0[:, -1:].T      

        if args.aug_num > 0 and not args.disable_filter:   # IPF
            weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)  # Δerase * C0 
            layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]   # > μ 
        else:
            layer_ret_embs = last_ret_embs

        sum_ret_ret, valid_num = [], 0
        for j in range(0, len(layer_ret_embs), chunk_size):   
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            if args.aug_num > 0:
                chunk_ret_embs = torch.cat(
                    [chunk_ret_embs, generate_perturbed_embs(chunk_ret_embs, P0_min, erase_weight, num_per_sample=args.aug_num)], dim=0
                )   # Rrefine = Rf + (Rf)^aug  
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))  # C0·C0^T
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num 

        if baseline == 'SPEED':
            U, S, V = torch.svd(sum_ret_ret)
            P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
            M = (sum_target_target @ P + args.retain_scale * I).inverse()
            
            base_delta_weight = layer_weight @ (((1 - alpha * target_embs @ anchor_embs.T / (anchor_embs.norm()**2))*sum_anchor_target) - (1 - alpha)*sum_target_target) @ P @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P) @ M
            
            adjusted_delta_weight = base_delta_weight

            if enable_semantic:
                try:
                    
                    if layer_idx % 4 == 0:  
                        current_diff = semantic_preserver.check_model_difference()
                        
                    semantic_losses = []
                    for i, (test_latent, test_timestep) in enumerate(zip(test_latents, test_timesteps)):
                        test_prompt = retain_texts[0] if retain_texts[0].strip() else "a photo"
                        
                        semantic_loss = semantic_preserver.get_semantic_preservation_loss(
                            test_prompt, test_latent, test_timestep
                        )
                        semantic_losses.append(semantic_loss)
                    
                    avg_semantic_loss = torch.stack(semantic_losses).mean()
                    semantic_constraint = layer_weight_factor * (args.semantic_lambda*1000.0) * avg_semantic_loss
                    adjustment_factor = torch.exp(-semantic_constraint)
                    adjusted_delta_weight = base_delta_weight * adjustment_factor
                    
                    semantic_applied_count += 1
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    adjusted_delta_weight = base_delta_weight


            if args.ace_enable:
                unconditional_delta = args.ace_lambda * layer_weight @ (-unconditional_target.T @ unconditional_target) @ P @ M
                delta_weight = adjusted_delta_weight + unconditional_delta
            else:
                delta_weight = adjusted_delta_weight

        # Save edited weights
        edit_dict[layer_name] = layer_weight + delta_weight
        module = pipeline.unet
        parts = layer_name.split('.')
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], torch.nn.Parameter(edit_dict[layer_name]))
    if enable_semantic:
        semantic_preserver.remove_hooks()
    
    print(f"Current model status: Edited {str(target_concepts)} into {str(anchor_concepts)}")
    return edit_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='CompVis/stable-diffusion-v1-4') 
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # Erase Config
    parser.add_argument('--target_concepts', type=str, required=True)
    parser.add_argument('--anchor_concepts', type=str, required=True)
    parser.add_argument('--retain_path', type=str, default=None)
    parser.add_argument('--heads', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='SPEED')
    # Hyperparameters
    parser.add_argument('--params', type=str, default='V')
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--retain_scale', type=float, default=1.0)
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--disable_filter', action='store_true', default=False)
    # Style preservation parameters
    parser.add_argument('--style_components', type=int, default=3, help='Number of style components to extract')
    parser.add_argument('--use_adaptive_threshold', action='store_true', help='Use adaptive threshold')
    # ACE parameters
    parser.add_argument('--ace_enable', action='store_true', default=True, help='Enable ACE anti-editing protection')
    parser.add_argument('--ace_lambda', type=float, default=0.3, help='Weight for ACE unconditional guidance')
    
    parser.add_argument('--semantic_lambda', type=float, default=3, help='Weight for semantic preservation loss')
    
    args = parser.parse_args()
    device = torch.device("cuda")
    seed_everything(args.seed)

    target_concepts = [con.strip() for con in args.target_concepts.split(',')] 
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path
    
    file_suffix = "_".join(target_concepts[:5])
    anchor_concepts = [x.strip() for x in anchor_concepts.split(',')]

    retain_texts = []
    if retain_path is not None:
        assert retain_path.endswith('.csv')
        df = pd.read_csv(retain_path)
        for head in args.heads.split(','):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")

    pipeline = StableDiffusionPipeline.from_pretrained(args.sd_ckpt).to(device)

    edit_dict = edit_model(
        args=args,
        pipeline=pipeline, 
        target_concepts=target_concepts, 
        anchor_concepts=anchor_concepts, 
        retain_texts=retain_texts, 
        baseline=args.baseline, 
        device=device, 
    )

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(edit_dict, os.path.join(save_path, f"{file_name}.pt"))