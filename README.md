# ComfyUI-SaveIntermediates

Save intermediate sampling steps as images during diffusion. Works with any sampler including ClownSampler, KSampler, etc.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DanielBartolic/ComfyUI-SaveIntermediates.git
```

## Usage

### Basic Usage

Connect the **Save Intermediate Steps** node between your model loader and sampler:

```
Load Checkpoint → [Save Intermediate Steps] → ClownSampler → VAE Decode → Save Image
```

### Nodes

#### Save Intermediate Steps
Basic node that saves every step.

**Inputs:**
- `model` - Connect from your model loader
- `output_folder` - Subfolder name in ComfyUI output directory (default: "intermediates")
- `filename_prefix` - Prefix for saved files (default: "step")
- `format` - Image format: jpeg or png
- `save_latents` - Also save raw latent tensors (.pt files)

**Output:**
- `model` - Pass-through to your sampler

#### Save Intermediate Steps (Advanced)
More control over which steps to save.

**Additional Inputs:**
- `save_every_n` - Save every Nth step (default: 1 = all steps)
- `start_step` - First step to save (default: 0)
- `end_step` - Last step to save (default: 999)

## Output

Images are saved to:
```
ComfyUI/output/intermediates/YYYYMMDD_HHMMSS/
├── step_0000.jpeg
├── step_0001.jpeg
├── step_0002.jpeg
...
└── latents/  (if save_latents enabled)
    ├── step_0000.pt
    └── ...
```

## How It Works

The node wraps your model and intercepts each call to `apply_model()` during sampling. It uses TAESD (Tiny AutoEncoder for Stable Diffusion) to quickly decode latents to preview images - the same method ComfyUI uses for live previews.

This means:
- Fast decoding (TAESD, not full VAE)
- Minimal overhead
- Works with any sampler
- No modification to samplers needed

## Requirements

- ComfyUI with TAESD models installed (usually automatic)
- Works with SD1.5, SDXL, SD3.5, Flux, etc.

## License

MIT
