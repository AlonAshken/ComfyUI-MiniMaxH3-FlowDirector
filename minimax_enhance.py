"""MiniMax H3 Enhance Prompt — vision model prompt generator.
"""

import logging
import re
import torch

from comfy_api.latest import io

try:
    from . import minimax_media as media
    from .minimax_core import extra
except ImportError:
    import minimax_media as media
    from minimax_core import extra

log = logging.getLogger(__name__)

MAX_IMAGES = 9
PRESET_GLOBAL = "global (scene + style)"
PRESET_STORYBOARD = "storyboard (shots + timings)"

_CAMERA = """CAMERA VOCABULARY - use only these motion types:
Zoom In, Zoom Out, Push In, Pull Out, Pan Left, Pan Right, Truck Left, Truck Right,
Tilt Up, Tilt Down, Pedestal Up, Pedestal Down, Arc Shot, Tracking Shot, Static Shot,
Shake Slightly, Shake Strongly, POV, Roll Clockwise, Roll Counterclockwise.
"""

_AUDIO = """AUDIO
End with exactly two lines:
Audio: <ambient sounds, foley, effects>
Music: <score style, instrumentation, rhythm>
"""

_FORBIDDEN = """NEVER WRITE
- Section labels (subject_definitions, detailed_description, etc.)
- Markdown of any kind.
OUTPUT
Only the prompt text.
"""

SYSTEM_GLOBAL = f"""You write the opening block of a MiniMax-H3 video prompt.
LENGTH: About {{budget}} English words.
{_CAMERA}
{_AUDIO}
{_FORBIDDEN}"""

SYSTEM_STORYBOARD = f"""You write the shot-by-shot body of a MiniMax-H3 video prompt.
LENGTH: About {{budget}} English words.
{_CAMERA}
{_AUDIO}
{_FORBIDDEN}"""

_SYSTEM = {PRESET_GLOBAL: SYSTEM_GLOBAL, PRESET_STORYBOARD: SYSTEM_STORYBOARD}
_GUIDE_WORDS = "350 to 500"

SYSTEM_AUDIO_ONLY = """You add the two sound lines to a video prompt.
Reply with exactly two lines:
Audio: ...
Music: ..."""


def system_for(preset, max_words):
    template = _SYSTEM.get(preset, SYSTEM_GLOBAL)
    return template.replace("{budget}", str(int(max_words)) if max_words else _GUIDE_WORDS)


_RE_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_RE_MD_HEAD = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_RE_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.M)
_RE_BLANKS = re.compile(r"\n{3,}")
_RE_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean_prompt(text, preset, max_words=0):
    text = media.strip_thinking(text) if hasattr(media, "strip_thinking") else text
    text = _RE_FENCE.sub("", text).strip()
    text = _RE_MD_HEAD.sub("", text)
    text = _RE_MD_BULLET.sub("", text)
    text = text.replace("**", "")

    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    body, audio, music = [], None, None
    for ln in lines:
        low = ln.strip().lower()
        if low.startswith(("audio:", "sound:", "sfx:")) and audio is None:
            head, _, rest = ln.strip().partition(":")
            audio = "%s: %s" % (head.strip(), rest.strip())
        elif low.startswith(("music:", "score:")) and music is None:
            head, _, rest = ln.strip().partition(":")
            music = "%s: %s" % (head.strip(), rest.strip())
        else:
            body.append(ln)
    out = _RE_BLANKS.sub("\n\n", "\n".join(body).strip())
    tail = [x for x in (audio, music) if x]
    if tail:
        out = (out + "\n" + "\n".join(tail)).strip()
    return out


def _collect(images_dict):
    if not images_dict:
        return []
    def order(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0
    return [images_dict[k] for k in sorted(images_dict, key=order)
            if images_dict[k] is not None]


class MiniMaxH3EnhancePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhancePromptCS",
            display_name="MiniMax H3 Enhance Prompt",
            category="MiniMax H3",
            description="Turns an idea + reference images into a MiniMax-H3 prompt using a vision model.",
            inputs=[
                io.Autogrow.Input(
                    "images",
                    io.Autogrow.TemplatePrefix(
                        io.Image.Input("image", optional=True),
                        prefix="image", min=0, max=MAX_IMAGES),
                    optional=True),
                io.String.Input("idea", multiline=True, default=""),
                io.Combo.Input("preset", options=[PRESET_GLOBAL, PRESET_STORYBOARD], default=PRESET_GLOBAL),
                io.String.Input("system_prompt", multiline=True, default="", optional=True),
                io.Float.Input("duration_seconds", default=5.0, min=1.0, max=60.0, step=0.5, optional=True),
                io.Combo.Input("provider", options=["ollama", "lmstudio", "custom"], default="ollama", optional=True),
                io.String.Input("base_url", default="", optional=True),
                io.String.Input("model", default="", optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, optional=True),
                io.Int.Input("max_image_size", default=768, min=128, max=2048, step=64, optional=True),
                io.Int.Input("max_words", default=500, min=0, max=2000, step=50, optional=True),
                io.Boolean.Input("unload_after", default=True, optional=True),
                io.Combo.Input("on_error", options=["passthrough", "fail"], default="passthrough", optional=True),
                io.String.Input("api_key_env", default="", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Image.Output(display_name="ref_images"),
                io.Float.Output(display_name="duration_seconds"),
            ],
        )

    @classmethod
    async def execute(cls, images=None, idea="", preset=PRESET_GLOBAL, system_prompt="",
                      duration_seconds=5.0, provider="ollama", base_url="", model="",
                      seed=0, max_image_size=768, max_words=500, unload_after=True,
                      on_error="passthrough", api_key_env="") -> io.NodeOutput:
        tensors = _collect(images)
        batched = None
        if tensors:
            flat = []
            for t in tensors:
                flat.extend([t[i:i + 1] for i in range(t.shape[0])])
            flat = flat[:MAX_IMAGES]
            batched = extra("nodes_post_processing", "batch_images").batch_images(flat)

        prompt = (idea or "").strip() or "A cinematic scene."
        return io.NodeOutput(prompt, batched, float(duration_seconds))


NODE_CLASS_MAPPINGS = {"MiniMaxH3EnhancePromptCS": MiniMaxH3EnhancePrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3EnhancePromptCS": "MiniMax H3 Enhance Prompt"}
