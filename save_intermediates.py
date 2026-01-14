"""
Save Intermediate Steps Node
Saves preview images at each sampling step for streaming to frontend
"""

import torch
import os
import json
import folder_paths
import comfy.model_management
from PIL import Image
import numpy as np
from typing import Optional
import time
import base64
from io import BytesIO


def get_previewer(model):
    """Get previewer for latent-to-image conversion"""
    try:
        import latent_preview
        device = comfy.model_management.get_torch_device()
        latent_format = model.model.latent_format
        previewer = latent_preview.get_previewer(device, latent_format)
        return previewer
    except Exception as e:
        print(f"[SaveIntermediates] Warning: Could not load previewer: {e}")
        return None


class IntermediateStreamer:
    """Handles saving/streaming intermediate latents as images"""

    def __init__(self, output_dir: str, prefix: str, format: str, previewer,
                 latent_format, job_id: str = "", save_base64: bool = False, total_steps: int = 20):
        self.output_dir = output_dir
        self.prefix = prefix
        self.format = format.lower()
        self.previewer = previewer
        self.latent_format = latent_format
        self.job_id = job_id
        self.save_base64 = save_base64
        self.total_steps = total_steps
        self.step_count = 0
        self.progress_file = os.path.join(output_dir, "progress.json")

        os.makedirs(output_dir, exist_ok=True)
        self._update_progress(0, None, "starting")

    def _update_progress(self, step: int, image_path: Optional[str], status: str, base64_data: str = None):
        """Update progress.json for frontend polling"""
        progress = {
            "job_id": self.job_id,
            "status": status,
            "current_step": step,
            "total_steps": self.total_steps,
            "progress_percent": int((step / max(self.total_steps, 1)) * 100),
            "latest_image": image_path,
            "timestamp": time.time(),
        }
        if base64_data and self.save_base64:
            progress["image_base64"] = base64_data

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            print(f"[SaveIntermediates] Error writing progress: {e}")

    def save_step(self, latent: torch.Tensor, step: int):
        """Save a single step's latent as an image"""
        try:
            if self.previewer is None:
                print(f"[SaveIntermediates] No previewer available")
                return None

            # Ensure proper shape: [B, C, H, W]
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            elif latent.dim() == 5:
                # Video latent [B, F, C, H, W] - take first frame
                latent = latent[:, 0, :, :, :]

            # Take first batch item, ensure float32, keep on same device
            latent_sample = latent[0:1].clone().to(torch.float32)

            # Process through latent format if available
            if self.latent_format is not None:
                try:
                    latent_sample = self.latent_format.process_out(latent_sample)
                except:
                    pass

            # Decode using previewer
            preview_result = self.previewer.decode_latent_to_preview_image(
                "JPEG" if self.format == "jpeg" else "PNG",
                latent_sample
            )

            if preview_result and len(preview_result) > 1:
                img = preview_result[1]

                filename = f"{self.prefix}_{step:04d}.{self.format}"
                filepath = os.path.join(self.output_dir, filename)
                img.save(filepath, quality=85 if self.format == "jpeg" else None)

                latest_path = os.path.join(self.output_dir, f"latest.{self.format}")
                img.save(latest_path, quality=85 if self.format == "jpeg" else None)

                base64_data = None
                if self.save_base64:
                    buffer = BytesIO()
                    img.save(buffer, format=self.format.upper(), quality=85)
                    base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

                self.step_count += 1
                self._update_progress(step, filename, "generating", base64_data)
                return filepath

        except Exception as e:
            print(f"[SaveIntermediates] Error saving step {step}: {e}")

        return None

    def finalize(self):
        """Mark generation as complete"""
        self._update_progress(self.total_steps, f"{self.prefix}_final.{self.format}", "completed")


class SaveIntermediateSteps:
    """
    Saves intermediate sampling steps for streaming to frontend.
    Works with Flux, SD3.5, SDXL, SD1.5, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
            },
            "optional": {
                "job_id": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": "progress"}),
                "filename_prefix": ("STRING", {"default": "step"}),
                "format": (["jpeg", "png"], {"default": "jpeg"}),
                "include_base64": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "wrap_model"
    CATEGORY = "sampling/streaming"

    def wrap_model(self, model, steps: int, job_id: str = "", output_folder: str = "progress",
                   filename_prefix: str = "step", format: str = "jpeg", include_base64: bool = False):

        base_output = folder_paths.get_output_directory()
        if job_id:
            full_output_dir = os.path.join(base_output, output_folder, job_id)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            full_output_dir = os.path.join(base_output, output_folder, timestamp)

        previewer = get_previewer(model)
        latent_format = model.model.latent_format if hasattr(model.model, 'latent_format') else None

        streamer = IntermediateStreamer(
            output_dir=full_output_dir,
            prefix=filename_prefix,
            format=format,
            previewer=previewer,
            latent_format=latent_format,
            job_id=job_id,
            save_base64=include_base64,
            total_steps=steps
        )

        # Clone model
        wrapped_model = model.clone()

        # Check model type
        model_sampling = wrapped_model.model.model_sampling if hasattr(wrapped_model.model, 'model_sampling') else None
        is_flow_model = hasattr(model_sampling, 'calculate_denoised') if model_sampling else False

        # Get original apply_model
        original_apply_model = wrapped_model.model.apply_model
        step_counter = [0]

        def patched_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            # Call original
            model_output = original_apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

            try:
                # Calculate denoised based on model type
                if model_sampling is not None and is_flow_model:
                    # For Flux/Flow models - use the model's own denoising calculation
                    try:
                        sigma = model_sampling.sigma(t)
                        denoised = model_sampling.calculate_denoised(sigma, model_output, x)
                    except:
                        # Fallback for flow matching: denoised = x - (1-sigma) * v
                        sigma = model_sampling.sigma(t)
                        if sigma.dim() == 0:
                            sigma = sigma.unsqueeze(0)
                        while sigma.dim() < x.dim():
                            sigma = sigma.unsqueeze(-1)
                        # For rectified flow: x0 = x - sigma * v
                        denoised = x - sigma * model_output
                elif model_sampling is not None:
                    # For eps-prediction models (SD1.5, SDXL, etc.)
                    sigma = model_sampling.sigma(t)
                    if sigma.dim() == 0:
                        sigma = sigma.unsqueeze(0)
                    while sigma.dim() < x.dim():
                        sigma = sigma.unsqueeze(-1)
                    # x0 = x - sigma * eps
                    denoised = x - sigma * model_output
                else:
                    # Fallback: use x directly
                    denoised = x

                # Save the preview
                if denoised.dim() == 4:
                    streamer.save_step(denoised, step_counter[0])
                elif denoised.dim() == 5:
                    streamer.save_step(denoised[:, 0], step_counter[0])

                step_counter[0] += 1

            except Exception as e:
                print(f"[SaveIntermediates] Error at step {step_counter[0]}: {e}")
                step_counter[0] += 1

            return model_output

        wrapped_model.model.apply_model = patched_apply_model
        wrapped_model._intermediate_streamer = streamer

        print(f"[SaveIntermediates] Streaming to: {full_output_dir}")
        print(f"[SaveIntermediates] Poll progress at: {full_output_dir}/progress.json")
        print(f"[SaveIntermediates] Flow model detected: {is_flow_model}")

        return (wrapped_model,)


class SaveIntermediateStepsAdvanced:
    """
    Advanced version with step filtering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "save_every_n": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "job_id": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": "progress"}),
                "filename_prefix": ("STRING", {"default": "step"}),
                "format": (["jpeg", "png"], {"default": "jpeg"}),
                "include_base64": ("BOOLEAN", {"default": False}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "end_step": ("INT", {"default": 999, "min": 0, "max": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "wrap_model"
    CATEGORY = "sampling/streaming"

    def wrap_model(self, model, steps: int, save_every_n: int, job_id: str = "",
                   output_folder: str = "progress", filename_prefix: str = "step",
                   format: str = "jpeg", include_base64: bool = False,
                   start_step: int = 0, end_step: int = 999):

        base_output = folder_paths.get_output_directory()
        if job_id:
            full_output_dir = os.path.join(base_output, output_folder, job_id)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            full_output_dir = os.path.join(base_output, output_folder, timestamp)

        previewer = get_previewer(model)
        latent_format = model.model.latent_format if hasattr(model.model, 'latent_format') else None

        streamer = IntermediateStreamer(
            output_dir=full_output_dir,
            prefix=filename_prefix,
            format=format,
            previewer=previewer,
            latent_format=latent_format,
            job_id=job_id,
            save_base64=include_base64,
            total_steps=steps
        )

        wrapped_model = model.clone()
        model_sampling = wrapped_model.model.model_sampling if hasattr(wrapped_model.model, 'model_sampling') else None
        is_flow_model = hasattr(model_sampling, 'calculate_denoised') if model_sampling else False
        original_apply_model = wrapped_model.model.apply_model
        step_counter = [0]

        def patched_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            model_output = original_apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

            current_step = step_counter[0]
            should_save = (
                current_step >= start_step and
                current_step <= end_step and
                (current_step - start_step) % save_every_n == 0
            )

            if should_save:
                try:
                    if model_sampling is not None and is_flow_model:
                        try:
                            sigma = model_sampling.sigma(t)
                            denoised = model_sampling.calculate_denoised(sigma, model_output, x)
                        except:
                            sigma = model_sampling.sigma(t)
                            if sigma.dim() == 0:
                                sigma = sigma.unsqueeze(0)
                            while sigma.dim() < x.dim():
                                sigma = sigma.unsqueeze(-1)
                            denoised = x - sigma * model_output
                    elif model_sampling is not None:
                        sigma = model_sampling.sigma(t)
                        if sigma.dim() == 0:
                            sigma = sigma.unsqueeze(0)
                        while sigma.dim() < x.dim():
                            sigma = sigma.unsqueeze(-1)
                        denoised = x - sigma * model_output
                    else:
                        denoised = x

                    if denoised.dim() == 4:
                        streamer.save_step(denoised, current_step)
                    elif denoised.dim() == 5:
                        streamer.save_step(denoised[:, 0], current_step)

                except Exception as e:
                    print(f"[SaveIntermediates] Error at step {current_step}: {e}")

            step_counter[0] += 1
            return model_output

        wrapped_model.model.apply_model = patched_apply_model
        wrapped_model._intermediate_streamer = streamer

        print(f"[SaveIntermediates] Streaming to: {full_output_dir}")

        return (wrapped_model,)


class GetIntermediateImages:
    """
    Collects all intermediate images after sampling completes.
    Connect after your sampler to get all steps as IMAGE batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "job_id": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": "progress"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "collect_images"
    CATEGORY = "sampling/streaming"

    def collect_images(self, latent, job_id: str = "", output_folder: str = "progress"):
        """Collect all saved intermediate images as a batch"""

        base_output = folder_paths.get_output_directory()

        if job_id:
            search_dir = os.path.join(base_output, output_folder, job_id)
        else:
            progress_dir = os.path.join(base_output, output_folder)
            if os.path.exists(progress_dir):
                folders = sorted([f for f in os.listdir(progress_dir) if os.path.isdir(os.path.join(progress_dir, f))])
                if folders:
                    search_dir = os.path.join(progress_dir, folders[-1])
                else:
                    return (torch.zeros(1, 64, 64, 3),)
            else:
                return (torch.zeros(1, 64, 64, 3),)

        images = []
        if os.path.exists(search_dir):
            files = sorted([f for f in os.listdir(search_dir) if f.startswith("step_") and (f.endswith(".jpeg") or f.endswith(".png"))])

            for filename in files:
                filepath = os.path.join(search_dir, filename)
                try:
                    img = Image.open(filepath).convert("RGB")
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(torch.from_numpy(img_array))
                except Exception as e:
                    print(f"[GetIntermediateImages] Error loading {filename}: {e}")

        if images:
            batch = torch.stack(images, dim=0)
            return (batch,)
        else:
            return (torch.zeros(1, 64, 64, 3),)
