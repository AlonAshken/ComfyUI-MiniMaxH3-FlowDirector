"""ComfyUI MiniMax H3 Flow Director Extension.
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
try:
    from .minimax_flow_director import MiniMaxH3FlowDirector
    from .minimax_enhance import MiniMaxH3EnhancePrompt
    from .minimax_lastframe import MiniMaxH3SaveLastFrame
    from .minimax_preview import MiniMaxH3PreviewOverride
except ImportError:
    from minimax_flow_director import MiniMaxH3FlowDirector
    from minimax_enhance import MiniMaxH3EnhancePrompt
    from minimax_lastframe import MiniMaxH3SaveLastFrame
    from minimax_preview import MiniMaxH3PreviewOverride


class MiniMaxH3FlowDirectorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MiniMaxH3FlowDirector,
            MiniMaxH3PreviewOverride,
            MiniMaxH3EnhancePrompt,
            MiniMaxH3SaveLastFrame,
        ]


async def comfy_entrypoint() -> MiniMaxH3FlowDirectorExtension:
    return MiniMaxH3FlowDirectorExtension()


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3FlowDirectorCS": MiniMaxH3FlowDirector,
    "MiniMaxH3PreviewOverrideCS": MiniMaxH3PreviewOverride,
    "MiniMaxH3EnhancePromptCS": MiniMaxH3EnhancePrompt,
    "MiniMaxH3SaveLastFrameCS": MiniMaxH3SaveLastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FlowDirectorCS": "MiniMax H3 Flow Director",
    "MiniMaxH3PreviewOverrideCS": "MiniMax H3 Preview Override",
    "MiniMaxH3EnhancePromptCS": "MiniMax H3 Enhance Prompt",
    "MiniMaxH3SaveLastFrameCS": "MiniMax H3 Save Last Frame",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
