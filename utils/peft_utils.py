# utils/peft_utils.py

import torch.nn as nn
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig

def wrap_with_lora(base_model, cfg: DictConfig):
    """
    Wraps a vanilla PyTorch model in LoRA adapters for Dual-Training.
    If LoRA is disabled in the config, it safely returns the base model.
    """
    lora_enabled = cfg.get("lora", {}).get("enabled", False)
    if not lora_enabled:
        return base_model
        
    # 1. Extract config params
    r = cfg.lora.get("r", 8)
    alpha = cfg.lora.get("alpha", 16)
    dropout = cfg.lora.get("dropout", 0.05)
    
    target_modules = []
    for name, module in base_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            target_modules.append(name)
            
    if not target_modules:
        raise ValueError("Could not find any nn.Linear or nn.Conv2d layers to wrap with LoRA!")
        
    # 2. Define the PEFT Config
    peft_config = LoraConfig(
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    # 3. Wrap the model
    peft_model = get_peft_model(base_model, peft_config)
    print("[Model] Wrapped Model in LoRA")
    
    # 4. EXPLICIT UNFREEZE: Allow Base Model to train alongside the Adapters
    for name, param in peft_model.base_model.named_parameters():
        param.requires_grad = True
        
    return peft_model
