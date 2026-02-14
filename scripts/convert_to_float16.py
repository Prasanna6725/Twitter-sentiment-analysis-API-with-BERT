#!/usr/bin/env python3
"""Quantize model to float16 to reduce file size"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import save_file
import torch
import os
import shutil
import json

os.makedirs('model_output_quantized', exist_ok=True)

print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained('model_output')

print('Converting to float16...')
model.half()

print('Saving quantized model...')
state_dict = model.state_dict()
save_file(state_dict, 'model_output_quantized/model.safetensors')

# Copy tokenizer files
print('Copying tokenizer files...')
shutil.copy('model_output/config.json', 'model_output_quantized/config.json')
shutil.copy('model_output/tokenizer.json', 'model_output_quantized/tokenizer.json')
shutil.copy('model_output/tokenizer_config.json', 'model_output_quantized/tokenizer_config.json')

# Update config to reflect float16
with open('model_output_quantized/config.json', 'r') as f:
    cfg = json.load(f)
cfg['torch_dtype'] = 'float16'
cfg['dtype'] = 'float16'
with open('model_output_quantized/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)

# Check size
size_mb = os.path.getsize('model_output_quantized/model.safetensors') / (1024**2)
print(f'\n✅ Done! New size: {size_mb:.2f} MB (was 255.43 MB)')
print(f'Size reduction: {100 * (1 - size_mb/255.43):.1f}%')
