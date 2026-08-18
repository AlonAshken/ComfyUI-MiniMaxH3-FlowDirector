# ComfyUI-MiniMaxH3-FlowDirector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)

**ComfyUI-MiniMaxH3-FlowDirector** is an advanced WYSIWYG visual timeline director for **MiniMax Hailuo H3**. It enables the generation of **infinitely long, continuous videos** with synchronized audio by breaking long timelines into discrete sequential flows, passing the decoded last frame as the first frame of subsequent blocks, and transitioning towards target images—all with a low, constant VRAM footprint.

---

## ✨ Key Features

1. **Discrete Sequential Flows (No OOM / No Crashes)**:
   - Instead of trying to sample a giant 1000-frame video all at once, each timeline segment is sampled and decoded in isolated blocks.
   - Low and constant memory footprint across any length (from 10 seconds to 10+ minutes).

2. **Automated Last-to-First Frame Chaining**:
   - The final decoded frame of Block $i$ is automatically passed as the opening keyframe (`first_frame`) to Block $i+1$, guaranteeing continuous motion flow.

3. **Target Transition Frames (`last_frame`)**:
   - Dragging an image onto Block $N > 0$ automatically treats that image as the **destination transition frame**. MiniMax H3 smoothly morphs and transitions from the previous block towards your target image!

4. **Seamless Seam Stitching**:
   - Redundant duplicate boundary frames between consecutive blocks are dropped automatically to ensure smooth playback without seam stutter.

5. **Integrated Audio & Motion Tracks**:
   - Full timeline audio waveforms with automatic trimming, stereo mixing, and native MiniMax audio decoding.

---

## 📦 Installation

### Method 1: Via Git Clone
Open your terminal in `ComfyUI/custom_nodes/` and run:

```bash
cd custom_nodes
git clone https://github.com/AlonAshken/ComfyUI-MiniMaxH3-FlowDirector.git
```

### Method 2: Via ComfyUI Manager
Search for `MiniMax H3 Flow Director` in ComfyUI Manager and click **Install**.

---

## 🚀 Recommended Workflow Integration

```
[FL2VA Model]        ───> [model       ]
[REF2VA Model]       ───> [model_ref2va]
[MiniMax Text CLIP]  ───> [clip        ]  [MiniMax H3 Flow Director]
[MiniMax Video VAE]  ───> [vae         ]  ─────────────────────────> images ───> [VHS Video Combine]
[MiniMax Audio VAE]  ───> [audio_vae   ]  ─────────────────────────> audio  ───> [                 ]
[KSamplerSelect]     ───> [sampler     ]  ─────────────────────────> fps    ───> [                 ]
[BasicScheduler]     ───> [sigmas      ]
[RandomNoise]        ───> [noise       ]
```

An example workflow is included in [`example_workflows/MINIMAX_H3_FLOW_DIRECTOR_ULTRA_TURBO.json`](example_workflows/MINIMAX_H3_FLOW_DIRECTOR_ULTRA_TURBO.json).

---

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.
