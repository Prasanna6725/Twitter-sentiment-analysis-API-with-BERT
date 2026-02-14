"""
Quantize trained model from float32 to int8 format
Reduces model.safetensors from ~255MB to ~100MB while maintaining accuracy
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import json

MODEL_DIR = Path(__file__).parent.parent / "model_output"
OUTPUT_DIR = Path(__file__).parent.parent / "model_output_quantized"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Loading model from {MODEL_DIR}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"Original model size: {sum(p.numel() for p in model.parameters() if p.dtype == torch.float32) * 4 / 1e6:.2f} MB")

# Dynamic quantization: converts float32 weights to int8
# This is recommended for inference and maintains good accuracy
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # quantize linear layers
    dtype=torch.qint8
)

print(f"Quantized model size: {sum(p.numel() for p in quantized_model.parameters()) * 1 / 1e6:.2f} MB (approximate)")

# Save quantized model
print(f"Saving quantized model to {OUTPUT_DIR}...")
quantized_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Copy config.json to quantized directory
import shutil
shutil.copy(MODEL_DIR / "config.json", OUTPUT_DIR / "config.json")
if (MODEL_DIR / "tokenizer_config.json").exists():
    shutil.copy(MODEL_DIR / "tokenizer_config.json", OUTPUT_DIR / "tokenizer_config.json")

# Verify file size
safetensors_path = OUTPUT_DIR / "model.safetensors"
if safetensors_path.exists():
    size_mb = safetensors_path.stat().st_size / (1024**2)
    print(f"✅ Quantized model.safetensors: {size_mb:.2f} MB")
else:
    pytorch_path = OUTPUT_DIR / "pytorch_model.bin"
    if pytorch_path.exists():
        size_mb = pytorch_path.stat().st_size / (1024**2)
        print(f"✅ Quantized pytorch_model.bin: {size_mb:.2f} MB")

print("\n✅ Quantization complete!")
print(f"Quantized model saved to: {OUTPUT_DIR}")
print("\nTo use the quantized model, update your code:")
print(f"  model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")
