"""
BiRefNet-BatchedBasedVideoRmbg

Standalone, self-contained ComfyUI node pack: BiRefNet background removal with
a batched GPU inference pipeline. Derived from the BiRefNet node of
1038lab/ComfyUI-RMBG (GPL-3.0). Registers its own node class key
("BiRefNetRMBG_Batched") so it can be installed alongside the original
ComfyUI-RMBG pack without collision. Shares the same model directory
(ComfyUI/models/RMBG/BiRefNet/), so weights are downloaded once.
"""

from .birefnet_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# No WEB_DIRECTORY: this pack ships no JS.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"### Loading: BiRefNet-BatchedBasedVideoRmbg ({len(NODE_CLASS_MAPPINGS)} node)")
