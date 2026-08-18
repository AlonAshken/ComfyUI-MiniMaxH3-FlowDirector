"""Media I/O, audio decoding/stitching, VLM character analysis, and HTTP server endpoints for MiniMax H3 Flow Director.
"""

import asyncio
import base64
import io as _io
import json
import logging
import math
import os
import platform
import subprocess
import wave

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger(__name__)

WORKSPACE_SUBDIR = "whatdreamscost"
AUDIO_SR = 44100
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts", ".wmv")


# --------------------------------------------------------------------------------------
# path helpers
# --------------------------------------------------------------------------------------

def resolve_input_path(rel_name: str):
    """Resolve a timeline file reference to an absolute path inside ComfyUI/input."""
    if not rel_name:
        return None
    input_dir = folder_paths.get_input_directory()
    candidates = [
        os.path.join(input_dir, rel_name),
        os.path.join(input_dir, WORKSPACE_SUBDIR, os.path.basename(rel_name)),
        os.path.join(input_dir, os.path.basename(rel_name)),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            return path
    return None


# --------------------------------------------------------------------------------------
# waveform peaks / audio extraction (timeline UI)
# --------------------------------------------------------------------------------------

def _peaks_from_int16(samples, num_peaks=200):
    peaks = []
    step = max(1, len(samples) // num_peaks)
    for i in range(num_peaks):
        chunk = samples[i * step:(i + 1) * step]
        peaks.append(float(np.max(np.abs(chunk)) / 32767.0) if len(chunk) else 0.0)
    return peaks


def read_wav_peaks(wav_path):
    with wave.open(wav_path, "rb") as w:
        n_frames = w.getnframes()
        if n_frames <= 0:
            return [0.0] * 200
        samples = np.frombuffer(w.readframes(n_frames), dtype=np.int16)
    return _peaks_from_int16(samples)


def _decode_to_int16_mono(container, rate):
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    audio_bytes = bytearray()
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            audio_bytes.extend(out.to_ndarray().tobytes())
    for out in resampler.resample(None):
        audio_bytes.extend(out.to_ndarray().tobytes())
    return audio_bytes


def extract_audio_from_video(video_path):
    """Write a mono 44.1 kHz WAV next to the video and return (relative path, peaks)."""
    try:
        base, _ = os.path.splitext(video_path)
        output_wav = base + "_extracted_audio.wav"

        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 44:
            peaks = read_wav_peaks(output_wav)
            rel_wav = os.path.relpath(output_wav, folder_paths.get_input_directory()).replace("\\", "/")
            return rel_wav, peaks

        with av.open(video_path) as container:
            if not container.streams.audio:
                return None, None
            audio_bytes = _decode_to_int16_mono(container, 44100)

        with wave.open(output_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(audio_bytes)

        peaks = read_wav_peaks(output_wav)
        rel_wav = os.path.relpath(output_wav, folder_paths.get_input_directory()).replace("\\", "/")
        return rel_wav, peaks
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] extract_audio_from_video error for %s: %s", video_path, e)
        return None, None


def get_audio_peaks(file_path):
    try:
        with av.open(file_path) as container:
            if not container.streams.audio:
                return [0.0] * 200
            audio_bytes = _decode_to_int16_mono(container, 44100)
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        return _peaks_from_int16(samples)
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] get_audio_peaks error for %s: %s", file_path, e)
        return [0.0] * 200


# --------------------------------------------------------------------------------------
# image / video loading and transformations
# --------------------------------------------------------------------------------------

def BLANK(h=768, w=1344):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_path_image_or_video(path: str, trim_sec: float = 0.0, fps: float = 24.0) -> torch.Tensor:
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        try:
            frames = load_video_tensor(path, trim_sec, 1.0 / max(1.0, float(fps)), out_fps=float(fps))
            if frames is not None and frames.shape[0] > 0:
                return frames[:1]
        except Exception as e:
            log.warning("[MiniMaxFlowDirector] Video frame decode failed for %s: %s", path, e)
    try:
        return _pil_to_tensor(Image.open(path))
    except Exception:
        # Fallback to PyAV if PIL cannot identify (e.g. video file with non-standard extension)
        try:
            frames = load_video_tensor(path, trim_sec, 1.0 / max(1.0, float(fps)), out_fps=float(fps))
            if frames is not None and frames.shape[0] > 0:
                return frames[:1]
        except Exception:
            pass
    return None


def load_image_source(b64_or_url: str, filename: str, fps: float = 24.0) -> torch.Tensor:
    """Load from either base64, a /view? URL, or a filename (image or video)."""
    if b64_or_url and b64_or_url.startswith("/view?"):
        try:
            q = parse_qs(urlparse(b64_or_url).query)
            fname = q.get("filename", [None])[0]
            subfolder = q.get("subfolder", [""])[0]
            if fname:
                path = os.path.join(folder_paths.get_input_directory(), subfolder, fname)
                t = _load_path_image_or_video(path, 0.0, fps)
                if t is not None:
                    return t
        except Exception as e:
            log.debug("[MiniMaxFlowDirector] URL parsing failed for %s: %s", b64_or_url, e)

    if filename:
        path = resolve_input_path(filename)
        if path:
            t = _load_path_image_or_video(path, 0.0, fps)
            if t is not None:
                return t

    if b64_or_url:
        try:
            b64_str = b64_or_url.split(",", 1)[1] if "," in b64_or_url else b64_or_url
            return _pil_to_tensor(Image.open(_io.BytesIO(base64.b64decode(b64_str))))
        except Exception as e:
            log.debug("[MiniMaxFlowDirector] Base64 decoding failed: %s", e)

    return BLANK()


def load_image_tensor(seg: dict, fps: float = 24.0) -> torch.Tensor:
    """Decode a timeline image or video segment's opening frame to [1, H, W, 3] float32 in [0, 1]."""
    file_ref = seg.get("imageFile") or seg.get("videoFile") or seg.get("fileName")
    trim_sec = float(seg.get("trimStart", 0)) / max(1.0, float(fps))
    if file_ref:
        path = resolve_input_path(file_ref)
        if path:
            t = _load_path_image_or_video(path, trim_sec, fps)
            if t is not None:
                return t

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return BLANK()
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return _pil_to_tensor(Image.open(_io.BytesIO(base64.b64decode(b64_str))))
    except Exception:
        return BLANK()


def _decode_target_size(width, height, max_short_edge, max_pixels):
    scale = min(1.0, max_short_edge / max(1, min(width, height)))
    if width * height * scale * scale > max_pixels:
        scale = math.sqrt(max_pixels / float(width * height))
    return max(16, int(round(width * scale)) // 2 * 2), max(16, int(round(height * scale)) // 2 * 2)


def load_video_tensor(file_ref: str, trim_start_sec: float, duration_sec: float,
                      out_fps: float = 24.0, max_short_edge: int = 768,
                      max_pixels: int = 768 * 1344) -> torch.Tensor:
    """Decode a video clip and resample it to out_fps, returning [N, H, W, 3] float32."""
    path = resolve_input_path(file_ref)
    if not path:
        log.warning("[MiniMaxFlowDirector] Video not found: %s", file_ref)
        return BLANK()

    n_out = max(1, int(round(duration_sec * out_fps)))
    targets = [trim_start_sec + i / out_fps for i in range(n_out)]

    frames = []
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            src_w = stream.width or stream.codec_context.width
            src_h = stream.height or stream.codec_context.height
            dst_w, dst_h = _decode_target_size(src_w, src_h, max_short_edge, max_pixels)

            seek_target = max(0.0, trim_start_sec - 0.5)
            if stream.time_base:
                container.seek(int(seek_target / float(stream.time_base)), stream=stream, backward=True)
            else:
                container.seek(int(seek_target * av.time_base), stream=stream, backward=True)

            idx = 0
            prev = None
            for frame in container.decode(stream):
                ftime = frame.time
                if ftime is None and frame.pts is not None and stream.time_base:
                    ftime = float(frame.pts * stream.time_base)
                if ftime is None:
                    ftime = 0.0

                while idx < n_out and ftime >= targets[idx] - 1e-6:
                    pick = frame
                    if prev is not None and abs(prev[0] - targets[idx]) < abs(ftime - targets[idx]):
                        pick = prev[1]
                    frames.append(pick.reformat(width=dst_w, height=dst_h, format="rgb24").to_ndarray())
                    idx += 1
                prev = (ftime, frame)
                if idx >= n_out:
                    break

            if frames and idx < n_out:
                frames.extend([frames[-1]] * (n_out - idx))
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] Video extract error for %s: %s", file_ref, e)

    if not frames:
        return BLANK()
    return torch.from_numpy(np.asarray(frames, dtype=np.float32) / 255.0)


def _decode_audio_stereo(buffer_or_path, target_sr=AUDIO_SR):
    blocks = []
    with av.open(buffer_or_path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                blocks.append(torch.from_numpy(out.to_ndarray()))
        for out in resampler.resample(None):
            blocks.append(torch.from_numpy(out.to_ndarray()))
    if not blocks:
        return None
    return torch.cat(blocks, dim=1)


def load_audio_segment(seg: dict, frame_rate: float, file_key: str = "audioFile"):
    """Load one timeline audio segment as a ComfyUI AUDIO dict."""
    waveform = None
    if seg.get(file_key):
        path = resolve_input_path(seg[file_key])
        if path:
            try:
                waveform = _decode_audio_stereo(path)
            except Exception as e:
                log.warning("[MiniMaxFlowDirector] Audio decode failed for %s: %s", seg.get(file_key), e)
    if waveform is None and seg.get("audioB64"):
        try:
            b64 = seg["audioB64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            waveform = _decode_audio_stereo(_io.BytesIO(base64.b64decode(b64)))
        except Exception as e:
            log.warning("[MiniMaxFlowDirector] Audio base64 decode failed: %s", e)
    if waveform is None:
        return None

    start = int(float(seg.get("trimStart", 0)) / frame_rate * AUDIO_SR)
    length = int(float(seg.get("length", 1)) / frame_rate * AUDIO_SR)
    clip = waveform[:, max(0, start):max(0, start) + max(1, length)]
    if clip.shape[1] <= 0:
        return None
    return {"waveform": clip.unsqueeze(0), "sample_rate": AUDIO_SR}


def resize_image(tensor: torch.Tensor, target_w: int, target_h: int,
                 method: str, divisible_by: int) -> torch.Tensor:
    """Resize [N, H, W, 3] to target canvas, snapping to divisible_by multiple."""
    def snap(val, div):
        return max(div, (int(val) // div) * div)

    def scaled(edge, ratio):
        return int(round(edge * ratio))

    tw, th = snap(target_w, divisible_by), snap(target_h, divisible_by)
    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor

    t = tensor.permute(0, 3, 1, 2)

    if method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        out = F.interpolate(t, size=(snap(scaled(H, ratio), divisible_by), snap(scaled(W, ratio), divisible_by)),
                            mode="bilinear", align_corners=False)
    elif method in ("pad", "pad green"):
        ratio = min(tw / W, th / H)
        new_w, new_h = snap(min(tw, scaled(W, ratio)), divisible_by), snap(min(th, scaled(H, ratio)), divisible_by)
        inner = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_l, pad_t = (tw - new_w) // 2, (th - new_h) // 2
        if method == "pad green":
            out = torch.zeros((N, C, th, tw), dtype=t.dtype, device=t.device)
            out[:, 0], out[:, 1], out[:, 2] = 102 / 255.0, 1.0, 0.0
            out[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = inner
        else:
            out = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t),
                        mode="constant", value=0)
    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w, new_h = max(tw, scaled(W, ratio)), max(th, scaled(H, ratio))
        inner = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left, top = (new_w - tw) // 2, (new_h - th) // 2
        out = inner[:, :, top:top + th, left:left + tw]
    else:
        out = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)

    return out.permute(0, 2, 3, 1)


def compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    if crf <= 0:
        return tensor
    N, H, W, C = tensor.shape
    h, w = (H // 2) * 2, (W // 2) * 2
    frames = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()
    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        for i in range(N):
            for pkt in stream.encode(av.VideoFrame.from_ndarray(frames[i], format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

        buf.seek(0)
        reader = av.open(buf, mode="r")
        decoded = [f.to_ndarray(format="rgb24") for f in reader.decode(video=0)]
        reader.close()
        if not decoded:
            return tensor

        out = tensor.clone()
        n = min(N, len(decoded))
        out[:n, :h, :w] = torch.from_numpy(
            np.stack(decoded).astype(np.float32) / 255.0)[:n].to(tensor.device, tensor.dtype)
        return out
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] img_compression failed: %s", e)
        return tensor


def build_combined_audio(timeline_data_str: str, start_frame: int, duration_frames: int,
                         frame_rate: float, override_audio: bool = False) -> dict:
    """Mix the timeline's audio track down to one AUDIO dict."""
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * AUDIO_SR)))
    empty = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32),
             "sample_rate": AUDIO_SR}

    if not timeline_data_str:
        return empty

    try:
        data = json.loads(timeline_data_str) if isinstance(timeline_data_str, str) else timeline_data_str
        if override_audio:
            audio_segs = data.get("motionSegments", [])
        else:
            audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty

    if not audio_segs:
        return empty

    out = torch.zeros((2, total_samples), dtype=torch.float32)
    file_key = "videoFile" if override_audio else "audioFile"

    for seg in audio_segs:
        buffer = None
        if seg.get(file_key):
            path = resolve_input_path(seg[file_key])
            if path:
                with open(path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        if not override_audio and buffer is None and seg.get("audioB64"):
            try:
                b64 = seg["audioB64"]
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                buffer = _io.BytesIO(base64.b64decode(b64))
            except Exception:
                pass
        if buffer is None:
            continue

        try:
            waveform = _decode_audio_stereo(buffer)
            if waveform is None:
                continue

            trim_start = float(seg.get("trimStart", 0))
            length = float(seg.get("length", 1))
            seg_start = float(seg.get("start", 0))

            if seg_start + length <= start_frame:
                continue
            offset = max(0, start_frame - seg_start)
            trim_start += offset
            length = max(1, length - offset)
            seg_start = max(0, seg_start - start_frame)

            src_a = max(0, int(trim_start / frame_rate * AUDIO_SR))
            src_b = min(waveform.shape[1], src_a + int(length / frame_rate * AUDIO_SR))
            n = src_b - src_a
            if n <= 0:
                continue

            dst_a = int(seg_start / frame_rate * AUDIO_SR)
            if dst_a >= out.shape[1]:
                continue
            n = min(n, out.shape[1] - dst_a)
            if n <= 0:
                continue

            out[:, dst_a:dst_a + n] += waveform[:, src_a:src_a + n]
        except Exception as e:
            log.warning("[MiniMaxFlowDirector] Audio mix error: %s", e)
            continue

    return {"waveform": out.unsqueeze(0), "sample_rate": AUDIO_SR}


# --------------------------------------------------------------------------------------
# VLM Character Analysis
# --------------------------------------------------------------------------------------

_PROVIDER_DEFAULTS = {
    "ollama": {"url": "http://127.0.0.1:11434", "model": "qwen2.5vl:7b"},
    "lmstudio": {"url": "http://127.0.0.1:1234", "model": ""},
    "custom": {"url": "", "model": ""},
}

API_KEY_ENV_VARS = ("MINIMAX_DIRECTOR_VLM_API_KEY", "OPENAI_API_KEY")


def normalize_base_url(url, fallback=""):
    url = (url or "").strip().rstrip("/")
    if not url:
        return (fallback or "").strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    return url


def resolve_api_key(data=None):
    data = data or {}
    key = (data.get("api_key") or "").strip()
    if key:
        return key
    named = (data.get("api_key_env") or "").strip()
    for var in ([named] if named else []) + list(API_KEY_ENV_VARS):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return ""


def _auth_headers(api_key):
    key = (api_key or "").strip()
    return {"Authorization": "Bearer %s" % key} if key else None


class VLMError(RuntimeError):
    pass


def strip_thinking(text):
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif text.startswith("<think>"):
        text = ""
    return text


async def vlm_generate(images_b64, prompt, provider, base_url, model,
                       system_prompt=None, timeout=120, keep_alive=0, max_tokens=None,
                       api_key=None):
    import aiohttp

    if provider in ("lmstudio", "custom") and not model:
        raise VLMError("No model name set for %s." % provider)

    headers = _auth_headers(api_key)

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            if provider == "ollama":
                payload = {"model": model, "prompt": prompt, "images": images_b64,
                           "stream": False, "keep_alive": keep_alive, "think": False}
                if system_prompt:
                    payload["system"] = system_prompt
                if max_tokens:
                    payload["options"] = {"num_predict": int(max_tokens)}
                async with session.post("%s/api/generate" % base_url, json=payload,
                                        timeout=timeout) as response:
                    if response.status != 200:
                        raise VLMError("Ollama HTTP %s: %s" % (response.status, await response.text()))
                    body = await response.json()
                    text = ((body.get("response") or "").strip()
                            or (body.get("thinking") or "").strip())
            else:
                content = [{"type": "text", "text": prompt}]
                for b64 in images_b64:
                    content.append({"type": "image_url",
                                    "image_url": {"url": "data:image/jpeg;base64,%s" % b64}})
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": content})
                payload = {"model": model, "messages": messages,
                           "max_tokens": int(max_tokens) if max_tokens else 2048,
                           "stream": False}
                async with session.post("%s/v1/chat/completions" % base_url, json=payload,
                                        timeout=timeout) as response:
                    if response.status != 200:
                        raise VLMError("%s HTTP %s: %s" % (provider, response.status, await response.text()))
                    resp_json = await response.json()
                    msg = resp_json["choices"][0]["message"]
                    text = (msg.get("content") or "").strip()
                    if not text:
                        text = (msg.get("reasoning_content") or "").strip()
    except Exception as e:
        raise VLMError(str(e))

    return strip_thinking(text)


async def unload_model(provider, base_url, model, api_key=None):
    if not model:
        return False
    try:
        import aiohttp
    except Exception:
        return False

    headers = _auth_headers(api_key)
    if provider == "ollama":
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post("%s/api/generate" % base_url,
                                        json={"model": model, "keep_alive": 0},
                                        timeout=10) as response:
                    await response.text()
            return True
        except Exception:
            return False
    return False


def _route(method, path):
    def decorator(fn):
        instance = getattr(PromptServer, "instance", None)
        if instance is not None and getattr(instance, "routes", None) is not None:
            try:
                getattr(instance.routes, method)(path)(fn)
            except Exception:
                pass
        return fn
    return decorator


@_route("get", "/minimax_flow_director_check_file")
async def flow_director_check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})

    upload_dir = folder_paths.get_input_directory()
    temp_dir = os.path.join(upload_dir, WORKSPACE_SUBDIR)

    def _matches(path):
        if not (os.path.exists(path) and os.path.isfile(path)):
            return False
        if not file_size:
            return True
        try:
            return os.path.getsize(path) == int(file_size)
        except ValueError:
            return True

    for path in (os.path.join(temp_dir, filename), os.path.join(upload_dir, filename)):
        if _matches(path):
            rel = os.path.relpath(path, upload_dir).replace("\\", "/")
            return web.json_response({"exists": True, "name": rel})

    base_name = os.path.basename(filename)
    suffix = "_" + base_name
    try:
        for search_dir in (temp_dir, upload_dir):
            if not os.path.exists(search_dir):
                continue
            for f_name in os.listdir(search_dir):
                if f_name.endswith(suffix) or f_name == base_name:
                    path = os.path.join(search_dir, f_name)
                    if _matches(path):
                        rel = os.path.relpath(path, upload_dir).replace("\\", "/")
                        return web.json_response({"exists": True, "name": rel})
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] Error listing input directory: %s", e)

    return web.json_response({"exists": False})


@_route("post", "/minimax_flow_director_upload_chunk")
async def flow_director_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = os.path.basename(post.get("filename"))
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = os.path.join(folder_paths.get_input_directory(), WORKSPACE_SUBDIR)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)

    if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
        return web.json_response({"error": "Invalid filename"}, status=400)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_chunk, file, file_path, "ab" if chunk_index > 0 else "wb")

    if chunk_index == total_chunks - 1:
        audio_file, peaks = None, None
        try:
            audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
        except Exception as e:
            log.warning("[MiniMaxFlowDirector] Error in final chunk audio extraction: %s", e)
        return web.json_response({
            "name": "%s/%s" % (WORKSPACE_SUBDIR, filename),
            "audio_file": audio_file,
            "peaks": peaks,
        })
    return web.json_response({"status": "ok"})


@_route("post", "/minimax_flow_director/compile_prompt")
async def compile_prompt_endpoint(request):
    """Live prompt preview endpoint for the flow director timeline."""
    try:
        from . import minimax_plan as plan
        data = await request.json()
        tdata = plan.parse_timeline(data.get("timeline_data"))
        fps = float(data.get("frame_rate") or 24.0)
        p = plan.plan_timeline(
            tdata=tdata,
            win_start=int(data.get("start_frame") or 0),
            duration_frames=int(data.get("duration_frames") or 120),
            fps=fps,
            global_prompt=data.get("global_prompt", ""),
            use_custom_motion=bool(data.get("use_custom_motion", True)),
            use_custom_audio=bool(data.get("use_custom_audio", False)),
            override_audio=bool(data.get("override_audio", False)),
            ref_image_notes=data.get("ref_image_notes", ""),
        )

        return web.json_response({
            "status": "success",
            "prompt": p["prompt"],
            "mode": p["mode"],
            "shots": len(p["shots"]),
            "length": p["length"],
            "seconds": round(p["actual_seconds"], 2),
            "blocks": len(p.get("blocks", [])),
            "refs": {"images": len(p.get("ref_image_slots", [])),
                     "videos": len(p.get("ref_video_segs", [])),
                     "audios": len(p.get("ref_audio_segs", []))},
        })
    except Exception as e:
        log.warning("[MiniMaxFlowDirector] compile_prompt failed: %s", e)
        return web.json_response({"status": "error", "message": str(e)}, status=500)


@_route("post", "/minimax_flow_director/probe_video")
async def probe_video_endpoint(request):
    try:
        data = await request.json()
        path = resolve_input_path(data.get("file") or "")
        if not path:
            return web.json_response({"status": "error", "message": "File not found on server."})

        with av.open(path) as container:
            if not container.streams.video:
                return web.json_response({"status": "error", "message": "No video stream in file."})
            stream = container.streams.video[0]
            codec = getattr(stream.codec_context, "name", "") or ""
            width = stream.width or stream.codec_context.width or 0
            height = stream.height or stream.codec_context.height or 0

            duration = 0.0
            if container.duration:
                duration = float(container.duration) / av.time_base
            elif stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)

            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else 0.0
            if duration <= 0 and fps > 0 and stream.frames:
                duration = float(stream.frames) / fps

            thumb = ""
            try:
                frame = next(container.decode(video=0), None)
                if frame is not None:
                    image = frame.to_image()
                    image.thumbnail((512, 512))
                    buffer = _io.BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=90)
                    thumb = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            except Exception:
                pass

        return web.json_response({"status": "success", "duration": round(duration, 3),
                                  "fps": round(fps, 4), "width": width, "height": height,
                                  "codec": codec, "thumb": thumb})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)})


@_route("get", "/minimax_flow_director_get_audio")
async def get_audio_endpoint(request):
    filename = request.query.get("filename")
    if not filename:
        return web.json_response({"error": "Missing filename"}, status=400)
    file_path = resolve_input_path(filename.replace("\\", "/"))
    if not file_path:
        return web.json_response({"error": "File not found"}, status=404)

    upload_dir = folder_paths.get_input_directory()
    if os.path.splitext(file_path)[1].lower() in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
        peaks = get_audio_peaks(file_path)
        rel = os.path.relpath(file_path, upload_dir).replace("\\", "/")
        return web.json_response({"audio_file": rel, "peaks": peaks})

    audio_file, peaks = extract_audio_from_video(file_path)
    return web.json_response({"audio_file": audio_file, "peaks": peaks})


@_route("get", "/minimax_flow_director_open_folder")
async def open_folder_endpoint(request):
    upload_dir = os.path.join(folder_paths.get_input_directory(), WORKSPACE_SUBDIR)
    os.makedirs(upload_dir, exist_ok=True)
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["explorer", os.path.normpath(upload_dir)])
        elif system == "Darwin":
            subprocess.Popen(["open", upload_dir])
        else:
            subprocess.Popen(["xdg-open", upload_dir])
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


@_route("post", "/minimax_flow_director/analyze_character")
async def analyze_character_endpoint(request):
    try:
        from . import minimax_plan as plan
        data = await request.json()
        image_b64 = data.get("image_b64", "")
        kind = plan.sanitize_kind(data.get("kind"))
        provider = (data.get("provider") or "ollama").lower()
        defs = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["ollama"])
        base_url = normalize_base_url(data.get("base_url"), defs["url"])
        model_name = data.get("model") or defs["model"]
        api_key = resolve_api_key(data)

        if provider == "off" or not image_b64:
            return web.json_response({"status": "error", "message": "Analyze off or no image."})

        b64_list = image_b64 if isinstance(image_b64, list) else [image_b64]
        cleaned = [b.split(",", 1)[1] if "," in b else b for b in b64_list]

        prompt = f"Describe the main {kind} in this image concisely."
        desc = await vlm_generate(cleaned, prompt, provider, base_url, model_name, api_key=api_key)
        return web.json_response({"status": "success", "description": desc, "retention_note": ""})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)})


@_route("post", "/minimax_flow_director/unload_ollama")
async def unload_ollama_endpoint(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    provider = (data.get("provider") or "ollama").lower()
    defs = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["ollama"])
    base_url = normalize_base_url(data.get("base_url"), defs["url"])
    model_name = data.get("model") or defs["model"]
    freed = await unload_model(provider, base_url, model_name, resolve_api_key(data))
    return web.json_response({"status": "ok", "released": freed})
