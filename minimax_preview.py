"""Live sampling preview for MiniMax H3.
"""

import base64
import io as _io
import logging
import struct
import time

import torch
import torch.nn.functional as F
from PIL import Image

import comfy.patcher_extension
import comfy.utils
import latent_preview
import server
from comfy_api.latest import io
from protocol import BinaryEventTypes

log = logging.getLogger(__name__)

EVENT = "minimax_h3_preview"
DECODE_FAST = "latent2rgb (fast)"
DECODE_VAE = "vae (quality)"
PLAYBACK_TRUE = "true speed"
PLAYBACK_SOURCE = "source fps"
MODEL_FPS = 24.0
TARGET_NODE = "node"
TARGET_SAMPLER = "sampler (VHS)"
TARGET_BOTH = "both"


def _video_stream(x0, latent_shapes=None):
    if x0 is None:
        return None
    if getattr(x0, "is_nested", False):
        return x0.tensors[0]
    if x0.ndim == 5:
        return x0
    if latent_shapes and len(latent_shapes) > 1:
        return comfy.utils.unpack_latents(x0, list(latent_shapes))[0]
    return None


def _pick_frames(video, max_frames):
    t = video.shape[2]
    if max_frames <= 0 or t <= max_frames:
        return video
    idx = torch.linspace(0, t - 1, max_frames).round().long().unique()
    return video[:, :, idx]


def pixel_frames_from_latent_t(latent_t):
    if latent_t <= 2:
        return 5
    k, remainder = divmod(int(latent_t) - 2, 5)
    if remainder == 0:
        return 17 * k + 5
    return max(1, int(round(latent_t * 17.0 / 5.0)))


class _RGBFactors:
    def __init__(self, latent_format):
        factors = getattr(latent_format, "latent_rgb_factors", None)
        if factors is None:
            raise ValueError("latent format has no latent_rgb_factors")
        self.w = torch.tensor(factors, device="cpu").transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.b = torch.tensor(bias, device="cpu") if bias is not None else None

    def __call__(self, video):
        chans = self.w.shape[1]
        moved = video.movedim(2, 1)
        x = moved.reshape((-1,) + tuple(moved.shape[-3:]))[:, :chans].to(torch.float32)
        w = self.w.to(dtype=x.dtype, device=x.device)
        b = self.b.to(dtype=x.dtype, device=x.device) if self.b is not None else None
        return ((F.linear(x.movedim(1, -1), w, bias=b) + 1.0) / 2.0).clamp(0, 1)


def _vae_decode(vae, video):
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    return images.clamp(0, 1).to(torch.float32).cpu()


def _to_pil(images, max_res):
    out = []
    for frame in images:
        arr = (frame * 255.0).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(arr)
        if max_res > 0:
            longest = max(img.width, img.height)
            if longest != max_res and longest > 0:
                scale = max_res / float(longest)
                size = (max(1, int(round(img.width * scale))),
                        max(1, int(round(img.height * scale))))
                img = img.resize(size, Image.LANCZOS if scale < 1.0 else Image.BICUBIC)
        out.append(img)
    return out


def _encode_animated_webp(frames, fps, quality):
    if not frames:
        return None
    buf = _io.BytesIO()
    try:
        frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:],
                       duration=max(1, int(round(1000 / max(1, fps)))), loop=0,
                       quality=quality, method=0)
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] animated WebP encode failed: %s", e)
        return None
    return base64.b64encode(buf.getvalue()).decode("ascii")


def throttle_gap(cost_seconds, max_overhead_percent):
    if max_overhead_percent <= 0 or cost_seconds <= 0:
        return 0.0
    return cost_seconds * (100.0 / float(max_overhead_percent) - 1.0)


class _VHSStreamer:
    def __init__(self, rate):
        self.rate = max(1, int(rate))
        self.first = True
        self.last_time = 0.0
        self.cursor = 0

    def send(self, pil_frames, rate=None):
        srv = server.PromptServer.instance
        total = len(pil_frames)
        if total == 0:
            return 0
        if rate and self.first:
            self.rate = max(1, int(round(rate)))
        now = time.time()
        count = int((now - self.last_time) * self.rate)
        self.last_time += count / self.rate
        if count > total:
            count = total
        elif count <= 0:
            return 0
        if self.first:
            self.first = False
            srv.send_sync("VHS_latentpreview",
                          {"length": total, "rate": self.rate, "id": srv.last_node_id})
            self.last_time = now + 1.0 / self.rate

        order = [(self.cursor + i) % total for i in range(count)]
        node_id = (srv.last_node_id or "").encode("ascii")
        for i in order:
            message = _io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(i.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", node_id))
            pil_frames[i].save(message, format="JPEG", quality=95)
            srv.send_sync(BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), srv.client_id)
        self.cursor = (self.cursor + count) % total
        return count


class _OuterSampleWrapper:
    def __init__(self, node_id, decode_mode, vae, max_resolution, preview_frames,
                 preview_fps, webp_quality, every_n_steps, suppress_default, target,
                 max_overhead=25, playback=None):
        self.node_id = node_id
        self.decode_mode = decode_mode
        self.vae = vae
        self.max_resolution = int(max_resolution)
        self.preview_frames = int(preview_frames)
        self.preview_fps = float(preview_fps)
        self.webp_quality = int(webp_quality)
        self.every_n_steps = max(1, int(every_n_steps))
        self.suppress_default = bool(suppress_default)
        self.target = target
        self.max_overhead = max(0, min(100, int(max_overhead)))
        self.playback = playback or PLAYBACK_TRUE

    def _rate_for(self, shown, pixel_frames):
        if self.playback == PLAYBACK_SOURCE:
            return max(0.1, self.preview_fps)
        return max(0.1, self.preview_fps * shown / max(1, pixel_frames))

    def _send_to_node(self, b64, n_frames, step, total_steps, ms, rate):
        server.PromptServer.instance.send_sync(EVENT, {
            "node_id": self.node_id, "webp": b64, "frames": n_frames,
            "fps": round(float(rate), 2), "source_fps": round(float(self.preview_fps), 2),
            "step": step, "total_steps": total_steps, "ms": ms, "mode": self.decode_mode,
            "playback": self.playback,
        })

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, **kwargs):
        guider = executor.class_obj
        latent_shapes = kwargs.get("latent_shapes")
        latent_format = guider.model_patcher.model.latent_format

        to_rgb = None
        if self.decode_mode == DECODE_FAST or self.vae is None:
            try:
                to_rgb = _RGBFactors(latent_format)
            except Exception as e:
                log.warning("[MiniMaxFlowDirector] preview unavailable: %s", e)

        vhs = _VHSStreamer(self.preview_fps) if self.target in (TARGET_SAMPLER, TARGET_BOTH) else None
        to_node = self.target in (TARGET_NODE, TARGET_BOTH)

        original_decode = latent_preview.LatentPreviewer.decode_latent_to_preview_image
        if self.suppress_default:
            latent_preview.LatentPreviewer.decode_latent_to_preview_image = lambda self_, preview_format, x0: None

        original_cb = callback
        state = {"warned": False, "sent": 0, "cost": 0.0, "finished": 0.0, "anim": 0.0}

        def _should_skip(now):
            gap = throttle_gap(state["cost"], self.max_overhead)
            if to_node:
                gap = max(gap, state["anim"])
            return gap > 0 and (now - state["finished"]) < gap

        def combined(step, x0, x, total_steps):
            if (to_rgb is not None or self.vae is not None) and x0 is not None \
                    and step % self.every_n_steps == 0 and not _should_skip(time.time()):
                t0 = time.time()
                try:
                    video = _video_stream(x0, latent_shapes)
                    if video is not None and video.ndim == 5:
                        pixel_frames = pixel_frames_from_latent_t(int(video.shape[2]))
                        video = _pick_frames(video, self.preview_frames)
                        if self.decode_mode == DECODE_VAE and self.vae is not None:
                            images = _vae_decode(self.vae, video)
                        else:
                            images = to_rgb(video)
                        frames = _to_pil(images, self.max_resolution)
                        rate = self._rate_for(len(frames), pixel_frames)
                        if to_node:
                            b64 = _encode_animated_webp(frames, rate, self.webp_quality)
                            if b64:
                                self._send_to_node(b64, len(frames), step + 1, total_steps,
                                                   int((time.time() - t0) * 1000), rate)
                        if vhs is not None:
                            vhs.send(frames, rate)
                        state["sent"] += len(frames)
                        state["cost"] = time.time() - t0
                        state["finished"] = time.time()
                        state["anim"] = len(frames) / max(0.1, rate) if to_node else 0.0
                except Exception as e:
                    if not state["warned"]:
                        state["warned"] = True
                        log.warning("[MiniMaxFlowDirector] preview failed: %r", e)
            if original_cb is not None:
                original_cb(step, x0, x, total_steps)

        try:
            out = executor(noise, latent_image, sampler, sigmas, denoise_mask, combined,
                           disable_pbar, seed, **kwargs)
        finally:
            latent_preview.LatentPreviewer.decode_latent_to_preview_image = original_decode
        return out


class MiniMaxH3PreviewOverride(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PreviewOverrideCS",
            display_name="MiniMax H3 Preview Override",
            category="MiniMax H3",
            description=(
                "Live preview of the whole shot while it denoises, shown on this node."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model to attach preview to."),
                io.Vae.Input("vae", optional=True, tooltip="minimax_h3_video_vae for vae quality decode."),
                io.Combo.Input("decode", options=[DECODE_FAST, DECODE_VAE], default=DECODE_FAST),
                io.Combo.Input("preview_target", options=[TARGET_NODE, TARGET_SAMPLER, TARGET_BOTH], default=TARGET_NODE),
                io.Int.Input("max_resolution", default=512, min=64, max=2048, step=32),
                io.Int.Input("preview_frames", default=24, min=1, max=512, step=1),
                io.Float.Input("preview_fps", default=24.0, min=1.0, max=60.0, step=1.0),
                io.Int.Input("webp_quality", default=80, min=1, max=100, step=1, optional=True),
                io.Int.Input("every_n_steps", default=1, min=1, max=50, step=1, optional=True),
                io.Int.Input("max_preview_overhead", default=25, min=0, max=100, step=5, optional=True),
                io.Boolean.Input("suppress_default_preview", default=True, optional=True),
                io.Combo.Input("playback", options=[PLAYBACK_TRUE, PLAYBACK_SOURCE], default=PLAYBACK_TRUE, optional=True),
            ],
            outputs=[io.Model.Output(tooltip="Model with preview attached.")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, decode=DECODE_FAST, preview_target=TARGET_NODE, max_resolution=512,
                preview_frames=24, preview_fps=24.0, playback=PLAYBACK_TRUE, webp_quality=80,
                every_n_steps=1, max_preview_overhead=25, suppress_default_preview=True,
                vae=None) -> io.NodeOutput:
        if decode == DECODE_VAE and vae is None:
            raise ValueError("MiniMax H3 Preview Override: decode='vae (quality)' requires a connected VAE.")

        m = model.clone()
        wrapper = _OuterSampleWrapper(
            str(cls.hidden.unique_id), decode, vae, max_resolution, preview_frames,
            preview_fps, webp_quality, every_n_steps, suppress_default_preview,
            preview_target, max_preview_overhead, playback)

        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "minimax_h3_preview", wrapper, m.model_options, is_model_options=True)

        registered = comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, m.model_options, is_model_options=True)
        if wrapper not in registered and hasattr(m, "add_wrapper_with_key"):
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   "minimax_h3_preview", wrapper)
        return io.NodeOutput(m)


NODE_CLASS_MAPPINGS = {"MiniMaxH3PreviewOverrideCS": MiniMaxH3PreviewOverride}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3PreviewOverrideCS": "MiniMax H3 Preview Override"}
