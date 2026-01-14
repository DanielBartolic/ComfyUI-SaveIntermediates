"""
Save Intermediate Steps Node
Wraps a model to save TAESD preview images at each sampling step
"""

import torch
import os
import folder_paths
import comfy.model_management
import comfy.utils
from PIL import Image
import numpy as np
from typing import Optional, Callable, Any
import time


def get_taesd_decoder(latent_format):
    """Get TAESD decoder for fast latent-to-image conversion"""
    try:
        import latent_preview

        device = comfy.model_management.get_torch_device()
        previewer = latent_preview.get_previewer(device, latent_format)
        return previewer
    except Exception as e:
        print(f"[SaveIntermediates] Warning: Could not load TAESD previewer: {e}")
        return None


class IntermediateSaver:
    """Handles saving intermediate latents as images"""

    def __init__(self, output_dir: str, prefix: str, format: str, previewer, save_latents: bool = False):
        self.output_dir = output_dir
        self.prefix = prefix
        self.format = format.lower()
        self.previewer = previewer
        self.save_latents = save_latents
        self.step_count = 0
        self.saved_images = []

        os.makedirs(output_dir, exist_ok=True)
        if save_latents:
            os.makedirs(os.path.join(output_dir, "latents"), exist_ok=True)

    def save_step(self, denoised: torch.Tensor, step: int, sigma: float = 0.0):
        """Save a single step's denoised latent as an image"""
        try:
            if self.previewer is None:
                return

            # Handle batch dimension
            if denoised.dim() == 4:
                latent = denoised[0:1]  # Take first in batch
            else:
                latent = denoised.unsqueeze(0)

            # Decode using TAESD
            preview_result = self.previewer.decode_latent_to_preview_image(
                self.format.upper(),
                latent
            )

            if preview_result and len(preview_result) > 1:
                img = preview_result[1]

                # Save image
                filename = f"{self.prefix}_{step:04d}"
                if sigma > 0:
                    filename += f"_s{sigma:.4f}"
                filepath = os.path.join(self.output_dir, f"{filename}.{self.format}")

                img.save(filepath, quality=95 if self.format == "jpeg" else None)
                self.saved_images.append(filepath)
                self.step_count += 1

            # Optionally save raw latent
            if self.save_latents:
                latent_path = os.path.join(self.output_dir, "latents", f"{self.prefix}_{step:04d}.pt")
                torch.save(denoised.cpu(), latent_path)

        except Exception as e:
            print(f"[SaveIntermediates] Error saving step {step}: {e}")


class SaveIntermediateSteps:
    """
    Wraps a model to save intermediate sampling steps as images.
    Connect between your model loader and sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "output_folder": ("STRING", {"default": "intermediates"}),
                "filename_prefix": ("STRING", {"default": "step"}),
                "format": (["jpeg", "png"], {"default": "jpeg"}),
            },
            "optional": {
                "save_latents": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "wrap_model"
    CATEGORY = "sampling/intermediate"

    def wrap_model(self, model, output_folder: str, filename_prefix: str, format: str, save_latents: bool = False):
        # Create output directory
        base_output = folder_paths.get_output_directory()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_output_dir = os.path.join(base_output, output_folder, timestamp)

        # Get TAESD previewer
        previewer = get_taesd_decoder(model.model.latent_format)

        # Create saver instance
        saver = IntermediateSaver(
            output_dir=full_output_dir,
            prefix=filename_prefix,
            format=format,
            previewer=previewer,
            save_latents=save_latents
        )

        # Clone model to avoid modifying original
        wrapped_model = model.clone()

        # Store original callback setter if exists
        original_set_model_sampler_cfg_function = None
        if hasattr(wrapped_model.model, 'set_model_sampler_cfg_function'):
            original_set_model_sampler_cfg_function = wrapped_model.model.set_model_sampler_cfg_function

        # Patch the model to intercept sampling
        original_apply_model = wrapped_model.model.apply_model

        step_counter = [0]  # Use list to allow modification in nested function

        def patched_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            result = original_apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

            # Calculate denoised estimate (x0 prediction)
            try:
                # For most models, we can estimate x0 from the model output
                if hasattr(wrapped_model.model, 'model_sampling'):
                    sigma = wrapped_model.model.model_sampling.sigma(t)
                    if sigma.dim() == 0:
                        sigma_val = sigma.item()
                    else:
                        sigma_val = sigma[0].item()
                else:
                    sigma_val = 0.0

                # Save the current noise prediction as intermediate
                # Note: This is eps, not x0. For preview purposes it still shows progress.
                saver.save_step(result, step_counter[0], sigma_val)
                step_counter[0] += 1

            except Exception as e:
                print(f"[SaveIntermediates] Error in patched_apply_model: {e}")

            return result

        wrapped_model.model.apply_model = patched_apply_model

        print(f"[SaveIntermediates] Will save intermediate steps to: {full_output_dir}")

        return (wrapped_model,)


class SaveIntermediateStepsAdvanced:
    """
    Advanced version with more control over which steps to save.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "output_folder": ("STRING", {"default": "intermediates"}),
                "filename_prefix": ("STRING", {"default": "step"}),
                "format": (["jpeg", "png"], {"default": "jpeg"}),
                "save_every_n": ("INT", {"default": 1, "min": 1, "max": 100}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "end_step": ("INT", {"default": 999, "min": 0, "max": 1000}),
            },
            "optional": {
                "save_latents": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "wrap_model"
    CATEGORY = "sampling/intermediate"

    def wrap_model(self, model, output_folder: str, filename_prefix: str, format: str,
                   save_every_n: int, start_step: int, end_step: int, save_latents: bool = False):

        base_output = folder_paths.get_output_directory()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_output_dir = os.path.join(base_output, output_folder, timestamp)

        previewer = get_taesd_decoder(model.model.latent_format)

        saver = IntermediateSaver(
            output_dir=full_output_dir,
            prefix=filename_prefix,
            format=format,
            previewer=previewer,
            save_latents=save_latents
        )

        wrapped_model = model.clone()
        original_apply_model = wrapped_model.model.apply_model

        step_counter = [0]

        def patched_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            result = original_apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

            current_step = step_counter[0]

            # Check if we should save this step
            should_save = (
                current_step >= start_step and
                current_step <= end_step and
                (current_step - start_step) % save_every_n == 0
            )

            if should_save:
                try:
                    if hasattr(wrapped_model.model, 'model_sampling'):
                        sigma = wrapped_model.model.model_sampling.sigma(t)
                        sigma_val = sigma[0].item() if sigma.dim() > 0 else sigma.item()
                    else:
                        sigma_val = 0.0

                    saver.save_step(result, current_step, sigma_val)

                except Exception as e:
                    print(f"[SaveIntermediates] Error saving step {current_step}: {e}")

            step_counter[0] += 1

            return result

        wrapped_model.model.apply_model = patched_apply_model

        print(f"[SaveIntermediates] Will save steps {start_step}-{end_step} (every {save_every_n}) to: {full_output_dir}")

        return (wrapped_model,)
