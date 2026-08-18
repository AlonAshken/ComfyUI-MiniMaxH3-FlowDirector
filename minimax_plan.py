"""Timeline -> MiniMax H3 block flow and storyboard planner.

Handles metadata interpretation, block segmentation, prompt compilation, character/subject
tag resolution, dialogue splitting, and reference mapping without touching pixels.
"""

import json
import logging
import re

log = logging.getLogger(__name__)

MODEL_FPS = 24.0

# MiniMax H3 Limits
MAX_REF_IMAGES = 9          # "<= 9 images"
MAX_REF_VIDEOS = 3          # "<= 3 clips"
MAX_REF_AUDIOS = 3          # "<= 3 clips"
MAX_REF_FILES = 12          # "at most 12 files in total across all input types"
REF_VIDEO_MIN_SEC = 2.0     # "each clip must be 2-15 seconds long"
REF_VIDEO_MAX_SEC = 15.0
REF_VIDEO_TOTAL_SEC = 15.0  # "total duration <= 15 seconds"
REF_VIDEO_SHORT_EDGE = 768
REF_VIDEO_ASPECT_BUDGET = 1344 / 768.0
REF_VIDEO_SIZES = (768, 640, 512, 384, 256)

# Output envelope per block
TRAINED_MIN_FRAMES = 96
TRAINED_MAX_FRAMES = 360

ROLE_FIRST = "first"
ROLE_LAST = "last"
ROLE_MIDDLE = "middle"

RETENTION_VISIBLE = ("fully_preserved", "partially_preserved",
                     "attribute_transfer", "weak_reference")
RETENTION_AUDIO = ("fully_copy", "partially_copy", "reference", "weak_reference")
RETENTION_DEFAULT = "fully_preserved"
RETENTION_AUDIO_DEFAULT = "reference"

SUBJECT_KINDS = {
    "person": "the person",
    "animal": "the animal",
    "object": "the object",
    "environment": "the environment",
    "clothing": "the outfit",
    "prop": "the prop",
    "interface": "the interface",
    "effect": "the visual effect",
    "style": "the visual style",
    "action": "the action",
    "expression": "the expression",
    "pose": "the pose",
}
SUBJECT_KIND_DEFAULT = "person"

RETAINED_DETAIL = {
    "person": "identity, face and clothing",
    "animal": "markings, coat and build",
    "object": "shape, colour and markings",
    "environment": "layout, furnishing and lighting",
    "clothing": "cut, colour and material",
    "prop": "shape, colour and wear",
    "interface": "layout, typography and iconography",
    "effect": "shape, colour and behaviour",
    "style": "palette, texture and grade",
    "action": "timing and body mechanics",
    "expression": "expression and gaze",
    "pose": "posture and limb placement",
    "picture": "framing and composition",
    "storyboard": "viewpoint, subject placement and shot order",
    "video": "camera work and motion",
}

REF_ROLE_AUTO = "auto"
REF_ROLE_STORYBOARD = "storyboard"
REF_ROLE_SUBJECT = "subject"
REF_ROLES = (REF_ROLE_AUTO, REF_ROLE_STORYBOARD, REF_ROLE_SUBJECT)

MAX_SUBJECT_SLOTS = 9

TASK_KEYFRAME = "keyframe completion"
TASK_REFERENCE = "reference generation"
TASK_EDITING = "video editing"
TASK_CONTINUATION = "video continuation"
TASK_AUDIO_REUSE = "audio reuse"
TASK_AUDIO_REFERENCE = "audio reference"
TASK_ORDER = (TASK_KEYFRAME, TASK_REFERENCE, TASK_EDITING, TASK_CONTINUATION,
              TASK_AUDIO_REUSE, TASK_AUDIO_REFERENCE)


def align_frame_count(n):
    """H3 only accepts frame counts on the 17k+5 grid."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def fmt_seconds(value):
    """0.0 -> '0s', 1.5 -> '1.5s'."""
    rounded = round(float(value), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return "%ds" % int(round(rounded))
    return "%.1fs" % rounded


def fmt_timestamp(seconds):
    """MM:SS.mmm — timestamp format."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return "%02d:%06.3f" % (minutes, rest)


CHAR_TAG_RE = re.compile(r"@(?:character|char|ref)(\d+)")


def substitute_char_tags(text, replacements, later=None, seen=None):
    """Swap @ref1/@character1/@char1 .. @ref9 for their resolved text."""
    if not text:
        return text or ""
    if seen is None:
        seen = set()

    def swap(match):
        slot = int(match.group(1))
        value = replacements.get(slot)
        if not value:
            return match.group(0)
        if later and slot in seen:
            value = later.get(slot) or value
        seen.add(slot)
        return value

    return CHAR_TAG_RE.sub(swap, text)


def sanitize_retention(value, audio=False):
    allowed = RETENTION_AUDIO if audio else RETENTION_VISIBLE
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return RETENTION_AUDIO_DEFAULT if audio else RETENTION_DEFAULT


def sanitize_kind(value):
    text = str(value or "").strip().lower()
    return text if text in SUBJECT_KINDS else SUBJECT_KIND_DEFAULT


def sanitize_ref_role(value):
    text = str(value or "").strip().lower()
    return text if text in REF_ROLES else REF_ROLE_AUTO


def overlaps(seg, win_start, win_end):
    start = float(seg.get("start", 0))
    length = float(seg.get("length", 1))
    return start < win_end and start + length > win_start


def parse_timeline(timeline_data):
    try:
        if isinstance(timeline_data, dict):
            return timeline_data
        return json.loads(timeline_data) if timeline_data else {}
    except Exception as e:
        log.error("[MiniMaxFlowDirector] timeline_data parse error: %s", e)
        return {}


def ref_mode_from(tdata):
    return str(tdata.get("reference_mode", "OFF")).upper() != "OFF"


def retake_state(tdata):
    if not tdata.get("retakeMode"):
        return None
    video = tdata.get("retakeVideo") or {}
    if not isinstance(video, dict):
        return None
    name = video.get("imageFile") or video.get("fileName")
    if not name:
        return None
    return {
        "video": name,
        "start": int(float(tdata.get("retakeStart", 0) or 0)),
        "length": max(1, int(float(tdata.get("retakeLength", 0) or 1))),
        "base_frames": int(float(video.get("videoDurationFrames", 0) or 0)),
        "prompt": tdata.get("retakePrompt", "") or "",
    }


FORMAT_MINIMAX = "minimax"
FORMAT_COMFYUI = "comfyui"


def retention_note(label, marker, kind=None, audio=False):
    if audio:
        return {
            "fully_copy": "%s is reused as the target video's complete final audio track.",
            "partially_copy": "part of %s is copied; other sounds are added, removed or replaced.",
            "reference": "the target follows %s without copying the original signal.",
            "weak_reference": "only a broad similarity to %s in category and atmosphere is kept.",
        }[marker] % label
    if marker == "fully_preserved":
        detail = RETAINED_DETAIL.get(kind or "")
        if detail:
            return "the %s of %s are retained." % (detail, label)
        return "%s is retained as defined." % label
    return {
        "partially_preserved": "%s is still used, with some of its defined characteristics changed.",
        "attribute_transfer": "the characteristics of %s are transferred to a different target subject.",
        "weak_reference": "only a broad similarity to %s in style, category, composition and atmosphere is kept.",
    }[marker] % label


# Dialogue splitting
_DIRECT_DIALOGUE_RE = re.compile(
    r"""(?mx)
    ^[ \t]*
    (?P<speaker>@ref\d+)
    (?:[ \t]+(?P<how>[^:\n]+?))?
    [ \t]*:[ \t]*
    (?P<words>[^\n]+)
    $
    """
)

NEAR_MISS_ALIAS_RE = re.compile(r"^[ \t]*@(character|char)(\d+)")


def split_dialogue(text):
    """Split shot text into prose description and spoken dialogue."""
    if not text:
        return "", []
    prose_lines = []
    dialogue_events = []
    for line in text.splitlines():
        m = _DIRECT_DIALOGUE_RE.match(line)
        if m:
            dialogue_events.append({
                "speaker_tag": m.group("speaker"),
                "how": (m.group("how") or "").strip(),
                "words": m.group("words").strip(),
            })
        else:
            prose_lines.append(line)
    return "\n".join(prose_lines).strip(), dialogue_events


def shot_has_text(shot):
    return bool((shot.get("prompt") or "").strip() or shot.get("dialogue"))


def compile_storyboard(global_prompt, shots, window_seconds):
    """Compile shot list into single flowing storyboard text."""
    parts = []
    if (global_prompt or "").strip():
        parts.append(global_prompt.strip())

    shot_entries = []
    for i, shot in enumerate(shots):
        text = (shot.get("prompt") or "").strip()
        spoken = shot.get("dialogue") or []
        lines = []
        if text:
            lines.append(text)
        for d in spoken:
            speaker = d.get("speaker_tag", "@ref1")
            how = (" " + d["how"]) if d.get("how") else ""
            lines.append("%s%s: %s" % (speaker, how, d.get("words", "")))
        if lines:
            shot_content = " ".join(lines)
            if len(shots) > 1:
                start_s = fmt_seconds(shot.get("start_sec", 0))
                end_s = fmt_seconds(shot.get("end_sec", window_seconds))
                shot_entries.append("[%s-%s] %s" % (start_s, end_s, shot_content))
            else:
                shot_entries.append(shot_content)

    if shot_entries:
        parts.append(" ".join(shot_entries))

    return "\n\n".join(parts).strip() if parts else "video"


# --------------------------------------------------------------------------------------
# Block Extraction for Sequential Generation Flow
# --------------------------------------------------------------------------------------

def extract_blocks(tdata, fps=24.0, default_duration_frames=120):
    """Extract individual main-track timeline segments as discrete generation Blocks.
    
    Each block represents an independent generation unit:
    - start_frame, length_frames, duration_seconds
    - segment data (imageFile, imageB64, prompt, etc.)
    - index on timeline
    """
    fps = max(1.0, float(fps))
    raw_segments = tdata.get("segments", []) or []
    
    # Filter valid main track segments
    segments = [s for s in raw_segments if float(s.get("length", 0)) > 0]
    segments.sort(key=lambda s: float(s.get("start", 0)))
    
    if not segments:
        # No segments placed on timeline: synthesize a single default block
        dur_f = align_frame_count(int(tdata.get("duration_frames") or default_duration_frames))
        return [{
            "index": 0,
            "start_frame": 0,
            "length_frames": dur_f,
            "duration_seconds": dur_f / fps,
            "prompt": tdata.get("global_prompt", ""),
            "segment": {},
            "kind": "text",
            "has_initial_image": False,
        }]

    blocks = []
    for idx, seg in enumerate(segments):
        start_f = int(float(seg.get("start", 0)))
        len_f = int(float(seg.get("length", 120)))
        # Snap length to MiniMax H3's 17k+5 frame grid
        aligned_len = align_frame_count(len_f)
        
        has_img = bool(seg.get("imageFile") or seg.get("imageB64"))
        kind = seg.get("type", "image")
        
        blocks.append({
            "index": idx,
            "start_frame": start_f,
            "length_frames": aligned_len,
            "raw_length_frames": len_f,
            "duration_seconds": aligned_len / fps,
            "prompt": seg.get("prompt", "") or "",
            "segment": seg,
            "kind": kind,
            "has_initial_image": has_img,
            "is_end_frame": bool(seg.get("isEndFrame")),
            "file_name": seg.get("fileName", ""),
        })

    return blocks


def plan_block(tdata, block, fps, global_prompt="", use_custom_motion=True,
               use_custom_audio=False, override_audio=False, ref_image_notes=""):
    """Plan prompt and references for a single discrete Block.
    
    For Block idx:
    - window covers [block['start_frame'], block['start_frame'] + block['length_frames']]
    - compiles local block prompt + global prompt + tags
    - loads active motion / audio references overlapping this block's timeframe
    """
    fps = max(1.0, float(fps))
    win_start = int(block.get("start_frame", 0))
    duration_frames = int(block.get("length_frames", 124))
    win_end = win_start + duration_frames
    window_seconds = duration_frames / fps
    
    ref_mode_on = ref_mode_from(tdata)
    
    # Subject slots
    subject_slots = []
    raw_slots = tdata.get("subjects")
    if not isinstance(raw_slots, list):
        raw_slots = tdata.get("characters", []) or []
    for info in raw_slots[:MAX_SUBJECT_SLOTS]:
        images_list = info.get("images", []) or []
        legacy_b64 = info.get("imageB64", "")
        if legacy_b64 and not images_list:
            images_list = [{"b64": legacy_b64, "name": info.get("fileName", "")}]
        subject_slots.append({
            "images": images_list,
            "description": info.get("description", "") or "",
            "short_name": (info.get("shortName", "") or "").strip(),
            "kind": sanitize_kind(info.get("kind")),
            "retention": sanitize_retention(info.get("retention")),
            "note": info.get("retentionNote", "") or "",
        })

    # Overlapping motion/audio segments for this specific block
    ref_video_segs = []
    ref_audio_segs = []
    if ref_mode_on:
        if use_custom_motion:
            motion = [s for s in (tdata.get("motionSegments", []) or [])
                      if s.get("videoFile") and overlaps(s, win_start, win_end)]
            motion.sort(key=lambda s: float(s.get("start", 0)))
            ref_video_segs = motion[:MAX_REF_VIDEOS]
        if use_custom_audio:
            audio = [s for s in (tdata.get("audioSegments", []) or [])
                     if (s.get("audioFile") or s.get("audioB64"))
                     and overlaps(s, win_start, win_end)]
            audio.sort(key=lambda s: float(s.get("start", 0)))
            ref_audio_segs = audio[:MAX_REF_AUDIOS]

    # Resolve character tags
    char_tag_values = {}
    char_tag_later = {}
    ref_image_slots = []
    
    if ref_mode_on:
        for slot_idx, slot in enumerate(subject_slots):
            if not slot["images"] or len(ref_image_slots) >= MAX_REF_IMAGES:
                continue
            for img in slot["images"]:
                if len(ref_image_slots) >= MAX_REF_IMAGES:
                    break
                ref_image_slots.append({"source": "char", "slot": slot_idx, "image": img})

        for slot_idx, slot in enumerate(subject_slots):
            ordinals = [i + 1 for i, s in enumerate(ref_image_slots)
                        if s.get("source") == "char" and s.get("slot") == slot_idx]
            if ordinals:
                char_tag_values[slot_idx + 1] = "<Picture %d>" % ordinals[0]
            elif slot["description"]:
                char_tag_values[slot_idx + 1] = slot["description"]
                if slot["short_name"]:
                    char_tag_later[slot_idx + 1] = slot["short_name"]
    else:
        for slot_idx, slot in enumerate(subject_slots):
            if slot["description"]:
                char_tag_values[slot_idx + 1] = slot["description"]
                if slot["short_name"]:
                    char_tag_later[slot_idx + 1] = slot["short_name"]

    # Block prompt compilation
    block_prompt_raw = block.get("prompt", "")
    global_p = (global_prompt or tdata.get("global_prompt", "") or "").strip()
    
    prose, spoken = split_dialogue(block_prompt_raw)
    
    seen_slots = set()
    prose_sub = substitute_char_tags(prose, char_tag_values, char_tag_later, seen_slots)
    global_sub = substitute_char_tags(global_p, char_tag_values, char_tag_later, seen_slots)
    
    shots = [{
        "prompt": prose_sub,
        "dialogue": spoken,
        "start_sec": 0.0,
        "end_sec": window_seconds,
    }]
    
    compiled = compile_storyboard(global_sub, shots, window_seconds)
    
    # Soundscape / music
    soundscape = (tdata.get("overall_soundscape", "") or "").strip()
    music = (tdata.get("non_diegetic_music", "") or "").strip()
    if soundscape:
        compiled += "\n\nAudio: " + soundscape
    if music:
        compiled += "\n\nMusic: " + music

    if not compiled.strip():
        compiled = "video"

    return {
        "block_index": block.get("index", 0),
        "prompt": compiled,
        "length": duration_frames,
        "actual_seconds": window_seconds,
        "win_start": win_start,
        "duration_frames": duration_frames,
        "ref_mode_on": ref_mode_on,
        "ref_image_slots": ref_image_slots,
        "ref_video_segs": ref_video_segs,
        "ref_audio_segs": ref_audio_segs,
        "shots": shots,
    }


def plan_timeline(tdata, win_start, duration_frames, fps, global_prompt="",
                   use_custom_motion=True, use_custom_audio=False, override_audio=False,
                   extra_ref_image_count=0, soundscape="", music="", prompt_format=None,
                   summary="", ref_image_notes=""):
    """Comprehensive planner across a window for live prompt preview & single-window mode."""
    blocks = extract_blocks(tdata, fps)
    if not blocks:
        duration_frames = align_frame_count(duration_frames)
        return {
            "prompt": global_prompt or "video",
            "mode": "t2va",
            "shots": [],
            "events": [],
            "retake": None,
            "ref_mode_on": False,
            "ref_image_slots": [],
            "ref_video_segs": [],
            "ref_audio_segs": [],
            "length": duration_frames,
            "actual_seconds": duration_frames / fps,
            "blocks": [],
        }

    # Plan each block
    block_plans = []
    for b in blocks:
        bp = plan_block(tdata, b, fps, global_prompt, use_custom_motion, use_custom_audio, override_audio)
        block_plans.append(bp)

    total_frames = sum(b["length_frames"] for b in blocks)
    # Total unique frames when stitched (dropping 1 frame on seams between blocks)
    stitched_frames = total_frames - (len(blocks) - 1) if len(blocks) > 1 else total_frames

    all_prompts = [f"[Block {i+1} | {b.get('length', 124)}f (~{b.get('actual_seconds', 5.0):.1f}s)] {b.get('prompt', '')}"
                   for i, b in enumerate(block_plans)]
    full_prompt_text = "\n\n".join(all_prompts)

    return {
        "prompt": full_prompt_text,
        "mode": "ref2va" if ref_mode_from(tdata) else "fl2va",
        "shots": [bp["shots"][0] for bp in block_plans if bp["shots"]],
        "events": [],
        "retake": None,
        "ref_mode_on": ref_mode_from(tdata),
        "ref_image_slots": block_plans[0]["ref_image_slots"] if block_plans else [],
        "ref_video_segs": block_plans[0]["ref_video_segs"] if block_plans else [],
        "ref_audio_segs": block_plans[0]["ref_audio_segs"] if block_plans else [],
        "length": stitched_frames,
        "actual_seconds": stitched_frames / fps,
        "blocks": block_plans,
        "block_count": len(blocks),
    }
