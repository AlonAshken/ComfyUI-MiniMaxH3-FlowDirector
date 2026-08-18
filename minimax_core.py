"""Access to ComfyUI's built-in comfy_extras node modules.

ComfyUI loads comfy_extras modules with `spec_from_file_location(<abs path minus
extension>, ...)`, so they land in sys.modules under their full filesystem path and there
is no importable `comfy_extras.nodes_minimax_h3` name. We look them up by path suffix
first — reusing the already-loaded instance, so we stay bit for bit identical to the nodes
the user sees in the menu — and only fall back to loading the file ourselves if that
lookup fails.
"""

import importlib.util
import logging
import os
import sys

log = logging.getLogger(__name__)

_CACHE = {}


def _find_in_sys_modules(module_name, probe_attr):
    suffix = "comfy_extras/" + module_name
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not str(name).replace("\\", "/").endswith(suffix):
            continue
        if probe_attr is None or hasattr(mod, probe_attr):
            return mod
    return None


def _load_from_file(module_name):
    import nodes as _comfy_nodes

    path = os.path.join(
        os.path.dirname(os.path.realpath(_comfy_nodes.__file__)),
        "comfy_extras", module_name + ".py",
    )
    if not os.path.exists(path):
        raise ImportError(
            "MiniMax H3 Flow Director: comfy_extras/%s.py not found at %s. Update ComfyUI."
            % (module_name, path)
        )
    spec = importlib.util.spec_from_file_location("_mmxflow_" + module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extra(module_name, probe_attr=None):
    """Return a comfy_extras module by name, loading it on first use."""
    if module_name not in _CACHE:
        mod = _find_in_sys_modules(module_name, probe_attr)
        if mod is None:
            mod = _load_from_file(module_name)
            log.info("[MiniMaxFlowDirector] Loaded comfy_extras/%s from file.", module_name)
        _CACHE[module_name] = mod
    return _CACHE[module_name]


def core():
    """The MiniMax H3 core node module (nodes_minimax_h3)."""
    return extra("nodes_minimax_h3", "MiniMaxH3ImageToVideo")


def samplers():
    """comfy_extras/nodes_custom_sampler.py — guiders, noise, SamplerCustomAdvanced."""
    return extra("nodes_custom_sampler", "SamplerCustomAdvanced")


def audio_nodes():
    """comfy_extras/nodes_audio.py — VAEDecodeAudio and friends."""
    return extra("nodes_audio", "VAEDecodeAudio")
