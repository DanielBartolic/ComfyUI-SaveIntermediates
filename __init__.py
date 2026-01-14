"""
ComfyUI-SaveIntermediates
Save/stream intermediate sampling steps as images during diffusion
"""

from .save_intermediates import SaveIntermediateSteps, SaveIntermediateStepsAdvanced, GetIntermediateImages

NODE_CLASS_MAPPINGS = {
    "SaveIntermediateSteps": SaveIntermediateSteps,
    "SaveIntermediateStepsAdvanced": SaveIntermediateStepsAdvanced,
    "GetIntermediateImages": GetIntermediateImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveIntermediateSteps": "Save Intermediate Steps",
    "SaveIntermediateStepsAdvanced": "Save Intermediate Steps (Advanced)",
    "GetIntermediateImages": "Get Intermediate Images",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
