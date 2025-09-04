# Flash Attention 2 Installation (Optional)

Flash Attention 2 can significantly improve inference speed but is optional. The API will automatically fall back to SDPA (Scaled Dot-Product Attention) if Flash Attention is not available.

## Status Check

The API will automatically detect if Flash Attention 2 is available and use the appropriate attention implementation:
- If available: Uses `flash_attention_2` for better performance
- If not available: Falls back to `sdpa` (still performant, built into PyTorch)

## Optional Installation

To install Flash Attention 2 for better performance:

```bash
# For CUDA 11.8
pip install flash-attn --no-build-isolation

# For CUDA 12.x
pip install flash-attn --no-build-isolation
```

### Requirements:
- CUDA 11.6 or higher
- PyTorch 2.0 or higher
- Linux operating system (not available on Windows/Mac)

### Troubleshooting

If you see the error:
```
FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed
```

This means:
1. The model was saved with Flash Attention enabled
2. But Flash Attention is not installed in your environment

**Solution**: The API now automatically falls back to SDPA, so this error should not occur. If it does, restart the API service.

## Performance Impact

- **With Flash Attention 2**: ~30-50% faster inference, lower memory usage
- **With SDPA (default)**: Good performance, built into PyTorch, no extra installation
- **Without either**: Falls back to vanilla attention (slowest)

## Verification

To check which attention implementation is being used, look at the startup logs:

```
INFO:api.core.voice_service:Flash Attention 2 is available, using it for better performance
```

or

```
INFO:api.core.voice_service:Flash Attention 2 not available, using sdpa attention
```