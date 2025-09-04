#!/usr/bin/env python3
"""Test script to check if the model loads correctly like in the demo."""

import sys
import os
sys.path.insert(0, '/home/bumpyclock/Projects/VibeVoice')

import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

print("Testing VibeVoice model loading...")

# Test with the same parameters as the demo
model_path = "microsoft/VibeVoice-1.5B"
device = "cuda"

try:
    print(f"Loading processor from {model_path}...")
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    print("✓ Processor loaded successfully")
    
    print(f"Loading model from {model_path}...")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'},
        attn_implementation='sdpa'  # Use sdpa to avoid flash_attn issues
    )
    print("✓ Model loaded successfully")
    
    # Set to eval mode
    model.eval()
    print("✓ Model set to eval mode")
    
    # Configure noise scheduler like the demo
    model.model.noise_scheduler = model.model.noise_scheduler.from_config(
        model.model.noise_scheduler.config,
        algorithm_type='sde-dpmsolver++',
        beta_schedule='squaredcos_cap_v2'
    )
    model.set_ddpm_inference_steps(num_steps=5)
    print("✓ Noise scheduler configured")
    
    # Test a simple generation setup
    test_script = "Speaker 0: Hello, this is a test."
    test_voice_samples = [[torch.zeros(24000, dtype=torch.float32)]]  # 1 second of silence
    
    print("\nPreparing test inputs...")
    inputs = processor(
        text=[test_script],
        voice_samples=test_voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Move to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to('cuda:0')
    
    print("✓ Inputs prepared and moved to device")
    
    print("\n✅ All model loading tests passed!")
    print("\nModel config attributes:")
    if hasattr(model.config, 'num_hidden_layers'):
        print(f"  - num_hidden_layers: {model.config.num_hidden_layers}")
    else:
        print("  - num_hidden_layers: NOT FOUND (this is the issue)")
    
    if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'config'):
        lm_config = model.model.language_model.config
        if hasattr(lm_config, 'num_hidden_layers'):
            print(f"  - language_model.config.num_hidden_layers: {lm_config.num_hidden_layers}")
    
    print("\nTrying a minimal generate call...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.3,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                refresh_negative=True
            )
        print("✓ Generate call successful!")
    except Exception as e:
        print(f"✗ Generate failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()