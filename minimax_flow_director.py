"""MiniMax H3 Flow Director — Block-by-block sequential video generator for long continuous videos with low VRAM.
"""

import gc
import json
import logging
import math

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

try:
    from . import minimax_media as media
    from . import minimax_plan as plan
    from .minimax_core import audio_nodes, core, samplers
except ImportError:
    import minimax_media as media
    import minimax_plan as plan
    from minimax_core import audio_nodes, core, samplers

log = logging.getLogger(__name__)

MODEL_FPS = plan.MODEL_FPS
DEFAULT_W, DEFAULT_H = 1344, 768
MIN_CANVAS_EDGE = 32


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _unpack(out):
    if out is None:
        return ()
    args = getattr(out, "args", None)
    if isinstance(args, (tuple, list)):
        return tuple(args)
    result = getattr(out, "result", None)
    if isinstance(result, (tuple, list)):
        return tuple(result)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    if isinstance(out, dict) and isinstance(out.get("result"), (tuple, list)):
        return tuple(out["result"])
    return (out,)


def _snap(value, multiple):
    return max(multiple, (int(value) // multiple) * multiple)


def resolve_canvas(mm, custom_width, custom_height, divisible_by, resize_method, first_image):
    div = max(1, int(divisible_by))
    if custom_width > 0 and custom_height > 0:
        if first_image is not None:
            fitted = media.resize_image(first_image[:1], custom_width, custom_height,
                                        resize_method, div)
            return int(fitted.shape[2]), int(fitted.shape[1])
        return _snap(custom_width, div), _snap(custom_height, div)

    if first_image is not None:
        src_h, src_w = int(first_image.shape[1]), int(first_image.shape[2])
    else:
        src_w, src_h = DEFAULT_W, DEFAULT_H

    if custom_width > 0:
        w = _snap(custom_width, div)
        return w, _snap(src_h * w / max(1, src_w), div)
    if custom_height > 0:
        h = _snap(custom_height, div)
        return _snap(src_w * h / max(1, src_h), div), h

    return mm.adapt_canvas(src_w, src_h)


def resolve_size(custom_width, custom_height, width=None, height=None):
    if width is not None and int(width) > 0:
        custom_width = int(width)
    if height is not None and int(height) > 0:
        custom_height = int(height)
    return int(custom_width), int(custom_height)


def _decode_video(vae, latent):
    import nodes as comfy_nodes
    samples_dict = latent if isinstance(latent, dict) else {"samples": latent}
    images = comfy_nodes.VAEDecode().decode(vae, samples_dict)[0]
    return images.to(torch.float32).cpu()


def _decode_audio(audio_vae, latent):
    samples_dict = latent if isinstance(latent, dict) else {"samples": latent}
    out = audio_nodes().VAEDecodeAudio.execute(vae=audio_vae, samples=samples_dict)
    args = getattr(out, "args", None) or getattr(out, "result", None) or (out,)
    return args[0]


def _audio_to_stereo(audio):
    if audio is None:
        return None, media.AUDIO_SR
    waveform = audio.get("waveform")
    if waveform is None:
        return None, media.AUDIO_SR
    if waveform.ndim == 3:
        waveform = waveform[0]
    waveform = waveform.to(torch.float32).cpu()
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform, int(audio.get("sample_rate", media.AUDIO_SR))


def load_ref_image_tensors(slots, fit, ref_images=None):
    tensors = []
    input_cursor = 0
    for slot in slots:
        source = slot.get("source")
        if source == "char":
            img = slot["image"]
            tensors.append(media.load_image_source(img.get("b64", ""), img.get("name", "")))
        elif source == "input":
            if ref_images is None:
                continue
            tensors.append(ref_images[input_cursor:input_cursor + 1])
            input_cursor += 1
        elif source == "timeline":
            tensor = slot.get("tensor")
            if tensor is not None:
                tensors.append(fit(tensor[:1]))
    return tensors


class _Unconnected:
    def __repr__(self):
        return "<unconnected>"


_UNCONNECTED = _Unconnected()


def pick_model(model_fl2va, model_ref2va, ref_mode_on):
    wanted, other = (model_ref2va, model_fl2va) if ref_mode_on else (model_fl2va, model_ref2va)
    label = "ref2va" if ref_mode_on else "fl2va"
    if wanted is not None:
        return wanted
    if other is not None:
        log.warning("[MiniMaxFlowDirector] The toolbar is on '%s' but no %s model is connected — "
                    "using the other one. Load minimax_h3_%s_* for optimal results.",
                    "Refs ON" if ref_mode_on else "Refs OFF", label, label)
        return other
    raise ValueError(
        "MiniMax H3 Flow Director: no model connected. Wire a UNETLoader into 'model (t2v/i2v)' "
        "(minimax_h3_fl2va_*) and/or 'model (ref2v)' (minimax_h3_ref2va_*)."
    )


# --------------------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------------------

class MiniMaxH3FlowDirector(io.ComfyNode):
    """Visual timeline flow director with block-by-block sequential generation and last-frame passing."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FlowDirectorCS",
            display_name="MiniMax H3 Flow Director",
            category="MiniMax H3",
            description=(
                "Visual timeline director for MiniMax H3. Each block on the timeline is "
                "generated sequentially as its own flow, automatically passing the decoded last "
                "frame to the next block for continuous long videos with low, constant VRAM usage."
            ),
            inputs=[
                io.Model.Input("model", display_name="model (t2v/i2v)", optional=True, lazy=True,
                               tooltip="fl2va weights (minimax_h3_fl2va_*), used when toolbar is 'Refs OFF'."),
                io.Model.Input("model_ref2va", display_name="model (ref2v)", optional=True, lazy=True,
                               tooltip="ref2va weights (minimax_h3_ref2va_*), used when toolbar is 'Refs ON'."),
                io.Clip.Input("clip", tooltip="MiniMax Qwen3-VL text encoder."),
                io.Vae.Input("vae", tooltip="minimax_h3_video_vae."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="minimax_h3_audio_vae for joint audio generation and decoding."),
                io.Sampler.Input("sampler", tooltip="Sampler from KSamplerSelect."),
                io.Sigmas.Input("sigmas", tooltip="Sigmas from BasicScheduler."),
                io.Noise.Input("noise", optional=True, tooltip="Noise from RandomNoise."),
                io.String.Input("global_prompt", multiline=True, default="", force_input=True, optional=True,
                                tooltip="Conditions all blocks: style, atmosphere, characters."),
                io.Float.Input("start_second", default=0.0, min=0.0, max=10000.0, step=0.01, optional=True),
                io.Float.Input("end_second", default=5.0, min=0.0, max=10000.0, step=0.01, optional=True),
                io.Float.Input("duration_seconds", default=5.0, min=0.1, max=10000.0, step=0.01, optional=True),
                io.Int.Input("start_frame", default=0, min=0, max=1000000, step=1, optional=True),
                io.Int.Input("end_frame", default=120, min=1, max=1000000, step=1, optional=True),
                io.Int.Input("duration_frames", default=120, min=1, max=1000000, step=1, optional=True),
                io.String.Input("timeline_data", default="",
                                tooltip="JSON state of the timeline editor (auto-managed)."),
                io.Boolean.Input("use_custom_audio", default=False, optional=True),
                io.Boolean.Input("use_custom_motion", default=True, optional=True),
                io.Boolean.Input("inpaint_audio", default=True, optional=True),
                io.String.Input("local_prompts", multiline=True, default="", optional=True),
                io.String.Input("segment_lengths", default="", optional=True),
                io.Float.Input("frame_rate", default=24, min=1, max=240, step=1, optional=True),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
                io.String.Input("guide_strength", default="", optional=True),
                io.Int.Input("custom_width", default=0, min=0, max=8192, step=1, optional=True),
                io.Int.Input("custom_height", default=0, min=0, max=8192, step=1, optional=True),
                io.Combo.Input("resize_method",
                               options=["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"],
                               default="crop", optional=True),
                io.Int.Input("divisible_by", default=32, min=1, max=256, step=1, optional=True),
                io.Int.Input("img_compression", default=0, min=0, max=100, step=1, optional=True),
                io.Boolean.Input("override_audio", default=False, optional=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", optional=True),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, optional=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, optional=True),
                io.Image.Input("ref_images", optional=True, tooltip="Extra reference images for ref2va."),
                io.String.Input("ref_image_notes", multiline=True, default="", optional=True),
                io.Float.Input("start", force_input=True, optional=True, default=0.0),
                io.Float.Input("end", force_input=True, optional=True, default=0.0),
                io.Float.Input("duration", force_input=True, optional=True, default=0.0),
                io.Int.Input("width", force_input=True, optional=True, default=0),
                io.Int.Input("height", force_input=True, optional=True, default=0),
                io.Image.Input("first_frame_override", optional=True,
                               tooltip="Optional external image to seed Block 0 (overriding timeline image)."),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="Patched model output."),
                io.Conditioning.Output(display_name="positive", tooltip="Conditioning output."),
                io.Latent.Output(display_name="latent", tooltip="Latent output."),
                io.Image.Output(display_name="images", tooltip="Full concatenated and decoded video frames (24 fps). Wire directly into VHS Video Combine."),
                io.Audio.Output(display_name="audio", tooltip="Full concatenated and decoded stereo audio waveform (44.1 kHz). Wire directly into VHS Video Combine."),
                io.Float.Output(display_name="fps", tooltip="Always 24.0 — MiniMax H3 native output rate. Wire into VHS Video Combine."),
                io.Int.Output(display_name="width", tooltip="Output canvas width."),
                io.Int.Output(display_name="height", tooltip="Output canvas height."),
                io.Int.Output(display_name="length", tooltip="Total rendered frame count."),
                io.String.Output(display_name="prompt", tooltip="The compiled storyboard prompt."),
                io.Image.Output(display_name="last_image", tooltip="Last frame of the final block."),
                io.String.Output(display_name="retake_info", tooltip="Retake window info."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, timeline_data="{}", model=_UNCONNECTED,
                          model_ref2va=_UNCONNECTED, **_):
        ref_on = plan.ref_mode_from(plan.parse_timeline(timeline_data))
        target = "model_ref2va" if ref_on else "model"
        have = {"model": model, "model_ref2va": model_ref2va}
        if have.get(target) is None and have.get(target) is not _UNCONNECTED:
            return [target]
        return []

    @classmethod
    @torch.inference_mode()
    def execute(cls, clip, vae, sampler, sigmas,
                model=None, model_ref2va=None, audio_vae=None, noise=None,
                global_prompt="",
                start_second=0.0, end_second=5.0, duration_seconds=5.0,
                start_frame=0, end_frame=120, duration_frames=120,
                timeline_data="",
                use_custom_audio=False, use_custom_motion=True, inpaint_audio=True,
                local_prompts="", segment_lengths="",
                frame_rate=24, display_mode="seconds", guide_strength="",
                custom_width=0, custom_height=0, resize_method="crop",
                divisible_by=32, img_compression=0,
                override_audio=False, ref_image_size="match",
                shift_video=12.0, shift_audio=3.0,
                ref_images=None, ref_image_notes="",
                start=0.0, end=0.0, duration=0.0,
                width=0, height=0,
                first_frame_override=None) -> io.NodeOutput:

        mm = core()
        sm = samplers()
        fps = float(frame_rate or 24.0)
        tdata = plan.parse_timeline(timeline_data)

        # 1. Extract discrete blocks
        blocks = plan.extract_blocks(tdata, fps, default_duration_frames=int(duration_frames))
        log.info("[MiniMaxFlowDirector] Starting sequential execution of %d block(s)...", len(blocks))

        # 2. Pick Model & Apply Sigma Shift
        ref_mode_on = plan.ref_mode_from(tdata)
        chosen_model = pick_model(model, model_ref2va, ref_mode_on)
        patched_model = _unpack(mm.MiniMaxH3SigmaShift.execute(
            model=chosen_model, shift_video=float(shift_video), shift_audio=float(shift_audio)
        ))[0]

        # 3. Resolve Canvas Geometry
        box_w, box_h = resolve_size(custom_width, custom_height, width, height)
        initial_sample_tensor = None
        if first_frame_override is not None:
            initial_sample_tensor = first_frame_override[:1]
        elif blocks and blocks[0].get("has_initial_image"):
            initial_sample_tensor = media.load_image_tensor(blocks[0]["segment"], fps=fps)

        canvas_w, canvas_h = resolve_canvas(
            mm, box_w, box_h, int(divisible_by), resize_method, initial_sample_tensor
        )

        if canvas_w < MIN_CANVAS_EDGE or canvas_h < MIN_CANVAS_EDGE:
            raise ValueError(
                "MiniMax H3 Flow Director: the canvas came out %dx%d. Minimum is %dpx."
                % (canvas_w, canvas_h, MIN_CANVAS_EDGE)
            )

        def fit_canvas(img_tensor):
            if img_tensor is None:
                return None
            out = media.resize_image(img_tensor, canvas_w, canvas_h, resize_method, int(divisible_by))
            if out.shape[1] != canvas_h or out.shape[2] != canvas_w:
                out = media.resize_image(out, canvas_w, canvas_h, "crop", int(divisible_by))
            if int(img_compression) > 0:
                out = media.compress_image(out, int(img_compression))
            return out

        # 4. Sampler, Sigmas & Noise Resolution
        chosen_sampler = sampler
        chosen_sigmas = sigmas

        # 5. Sequential Block Generation Loop
        prev_last_frame = fit_canvas(first_frame_override[:1]) if first_frame_override is not None else None
        all_block_images = []
        all_block_audio = []
        compiled_prompts_log = []
        first_positive = None
        last_sampled_latent = None

        pbar = comfy.utils.ProgressBar(len(blocks))

        for block_idx, block in enumerate(blocks):
            comfy.model_management.throw_exception_if_processing_interrupted()

            # Seed determination from input noise or default
            if noise is not None and hasattr(noise, "seed"):
                cur_seed = (int(noise.seed) + block_idx) & 0xffffffffffffffff
            else:
                cur_seed = (42 + block_idx * 10007) & 0xffffffffffffffff

            # Block Plan
            bp = plan.plan_block(
                tdata, block, fps,
                global_prompt=global_prompt,
                use_custom_motion=use_custom_motion,
                use_custom_audio=use_custom_audio,
                override_audio=override_audio,
                ref_image_notes=ref_image_notes,
            )
            block_prompt = bp["prompt"]
            block_len = bp["length"]

            # Keyframe Anchor Continuity:
            # - Block 0: uses explicit override, or block timeline image, or None (pure t2v).
            # - Block N > 0: AUTOMATICALLY opens on decoded last frame of Block N-1!
            #   AND if the user placed an image on Block N, it acts as the Target Transition Last Frame (last_frame)!
            if block_idx == 0:
                if prev_last_frame is not None:
                    start_frame_img = prev_last_frame
                elif block.get("has_initial_image"):
                    start_frame_img = fit_canvas(media.load_image_tensor(block["segment"], fps=fps))
                else:
                    start_frame_img = None
                end_frame_img = None
            else:
                start_frame_img = prev_last_frame
                # If an image was placed on this block, use it as the target last frame to transition towards!
                if block.get("has_initial_image"):
                    end_frame_img = fit_canvas(media.load_image_tensor(block["segment"], fps=fps))
                else:
                    end_frame_img = None

            prompt_header = f"[Block {block_idx + 1}/{len(blocks)} | {block_len}f (~{bp['actual_seconds']:.1f}s) | seed={cur_seed}]"
            if end_frame_img is not None:
                prompt_header += " [-> Target End Frame]"
            compiled_prompts_log.append(f"{prompt_header}\n{block_prompt}")
            log.info("[MiniMaxFlowDirector] Generating %s...", prompt_header)

            # Conditioning + Latent
            if bp["ref_mode_on"]:
                ref_imgs = load_ref_image_tensors(bp["ref_image_slots"], fit_canvas, ref_images)
                if start_frame_img is not None and len(ref_imgs) < plan.MAX_REF_IMAGES:
                    ref_imgs.insert(0, start_frame_img)
                if end_frame_img is not None and len(ref_imgs) < plan.MAX_REF_IMAGES:
                    ref_imgs.append(end_frame_img)

                ref_videos = {}
                for s_idx, seg in enumerate(bp["ref_video_segs"]):
                    v_frames = media.load_video_tensor(
                        seg["videoFile"], float(seg.get("trimStart", 0)) / fps, bp["actual_seconds"]
                    )
                    if v_frames.shape[0] >= 5:
                        ref_videos["ref_video_%d" % s_idx] = v_frames

                ref_audios = {}
                for a_idx, seg in enumerate(bp["ref_audio_segs"]):
                    a_chunk = media.load_audio_segment(seg, fps)
                    if a_chunk is not None:
                        ref_audios["ref_audio_%d" % a_idx] = a_chunk

                cond_out = mm.MiniMaxH3ReferenceToVideo.execute(
                    clip=clip, vae=vae, audio_vae=audio_vae, prompt=block_prompt,
                    width=canvas_w, height=canvas_h, length=block_len,
                    ref_image_size=ref_image_size,
                    ref_images={"ref_image_%d" % i: t for i, t in enumerate(ref_imgs)} or None,
                    ref_videos=ref_videos or None,
                    ref_audios=ref_audios or None,
                )
            else:
                cond_out = mm.MiniMaxH3ImageToVideo.execute(
                    clip=clip, vae=vae, prompt=block_prompt,
                    width=canvas_w, height=canvas_h, length=block_len,
                    first_frame=start_frame_img,
                    last_frame=end_frame_img
                )

            conditioning, latent = _unpack(cond_out)[:2]
            if first_positive is None:
                first_positive = conditioning

            # Free CLIP/text encoder from VRAM so DiT model gets 100% of 16 GB dedicated VRAM
            comfy.model_management.cleanup_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()

            # Sampling
            guider = _unpack(sm.BasicGuider.execute(model=patched_model, conditioning=conditioning))[0]
            block_noise = sm.Noise_RandomNoise(cur_seed)
            sampled = _unpack(sm.SamplerCustomAdvanced.execute(
                noise=block_noise, guider=guider, sampler=chosen_sampler, sigmas=chosen_sigmas, latent_image=latent
            ))[0]
            last_sampled_latent = sampled

            # Free DiT from GPU before VAE decoding
            del guider, block_noise, conditioning, latent
            comfy.model_management.cleanup_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()

            # Decode Video
            block_images = _decode_video(vae, sampled)

            # Decode Audio
            block_audio_waveform = None
            if audio_vae is not None:
                try:
                    audio_chunk, sr = _audio_to_stereo(_decode_audio(audio_vae, sampled))
                    if audio_chunk is not None:
                        if sr != media.AUDIO_SR:
                            try:
                                import torchaudio
                                audio_chunk = torchaudio.functional.resample(audio_chunk, sr, media.AUDIO_SR)
                            except Exception:
                                pass
                        block_audio_waveform = audio_chunk
                except Exception as e:
                    log.warning("[MiniMaxFlowDirector] Audio VAE decode failed: %s", e)

            # Move decoded frame tensor and audio to CPU RAM immediately to free VRAM
            block_images = block_images.cpu()
            if block_audio_waveform is not None:
                block_audio_waveform = block_audio_waveform.cpu()

            # Extract the actual decoded last frame for the next block
            prev_last_frame = block_images[-1:].clone()

            # Free sampled latent
            del sampled

            # Seamless seam stitching: drop frame 0 on blocks > 0 to prevent identical duplicate frame stutter
            if block_idx > 0 and block_images.shape[0] > 1:
                block_images = block_images[1:]
                if block_audio_waveform is not None:
                    cut = int(round(1.0 / MODEL_FPS * media.AUDIO_SR))
                    block_audio_waveform = block_audio_waveform[:, cut:]

            all_block_images.append(block_images)
            if block_audio_waveform is not None:
                all_block_audio.append(block_audio_waveform)

            # Clean VRAM memory between blocks to keep consumption minimal!
            comfy.model_management.cleanup_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()

            log.info("[MiniMaxFlowDirector] Block %d/%d done: %d frames.",
                     block_idx + 1, len(blocks), block_images.shape[0])
            pbar.update(1)

        # 6. Concatenate full video & audio
        out_images = torch.cat(all_block_images, dim=0)

        if all_block_audio:
            out_audio_wave = torch.cat(all_block_audio, dim=1)
        else:
            mixed = media.build_combined_audio(
                timeline_data, 0, int(round(out_images.shape[0])), fps, override_audio=override_audio
            )
            out_audio_wave = mixed["waveform"]
            if out_audio_wave.ndim == 3:
                out_audio_wave = out_audio_wave[0]

        final_audio = {"waveform": out_audio_wave.unsqueeze(0), "sample_rate": media.AUDIO_SR}
        final_prompts = "\n\n".join(compiled_prompts_log)
        last_image_out = prev_last_frame if prev_last_frame is not None else out_images[-1:]

        return io.NodeOutput(
            patched_model,
            first_positive,
            last_sampled_latent,
            out_images,
            final_audio,
            float(fps),
            int(canvas_w),
            int(canvas_h),
            int(out_images.shape[0]),
            final_prompts,
            last_image_out,
            "",
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3FlowDirectorCS": MiniMaxH3FlowDirector}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3FlowDirectorCS": "MiniMax H3 Flow Director"}
