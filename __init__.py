"""
ComfyUI-SaveIntermediates
Save intermediate sampling steps as images during diffusion
"""

from .save_intermediates import SaveIntermediateSteps, SaveIntermediateStepsAdvanced

NODE_CLASS_MAPPINGS = {
    "SaveIntermediateSteps": SaveIntermediateSteps,
    "SaveIntermediateStepsAdvanced": SaveIntermediateStepsAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveIntermediateSteps": "Save Intermediate Steps",
    "SaveIntermediateStepsAdvanced": "Save Intermediate Steps (Advanced)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
