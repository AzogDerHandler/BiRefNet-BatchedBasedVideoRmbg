# BiRefNet-BatchedBasedVideoRmbg
# Standalone ComfyUI node: batched-GPU BiRefNet background removal.
#
# Derived from the BiRefNet node of 1038lab/ComfyUI-RMBG (GPL-3.0), with the
# per-image, PIL-based inference loop replaced by a true batched GPU pipeline
# that keeps everything as device tensors end to end. Inference runs on the GPU
# with a single device->host transfer per batch; the only remaining CPU/PIL step
# is the final mask blur/offset on the one-channel output mask (for exact parity
# with the stock edge feathering).
#
# This pack is self-contained and registers its own node class key
# ("BiRefNetRMBG_Batched"), so it can be installed ALONGSIDE the original
# ComfyUI-RMBG pack without collision. Both packs share the same model
# directory (ComfyUI/models/RMBG/BiRefNet/), so weights are downloaded once.
#
# Model License Notice:
# - BiRefNet Models: Apache-2.0 License (https://huggingface.co/ZhengPeng7)
#
# This integration script follows GPL-3.0 License (see LICENSE).
# When using or modifying this code, please respect both the original model
# licenses and this integration's license terms.
#
# Upstream source: https://github.com/1038lab/ComfyUI-RMBG

import os
import sys
import types
import importlib.util

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import folder_paths
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import cv2

# Respect ComfyUI's device selection (--cpu, --cuda-device, MPS, ...) instead of
# hardcoding "cuda if available". Fall back to the naive pick only when ComfyUI's
# model management isn't importable (e.g. running the smoke test standalone).
try:
    from comfy.model_management import get_torch_device
    device = get_torch_device()
except Exception:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Note: deliberately no torch.set_float32_matmul_precision(...) here — that is
# process-wide global state and would silently switch TF32 on for every other
# node's fp32 matmuls. Upstream sets it at model load; this pack does not.

# Add model path
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# ImageNet normalisation stats (BiRefNet was trained with these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model configuration
MODEL_CONFIG = {
    "BiRefNet-general": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-general.safetensors": "BiRefNet-general.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "General purpose model with balanced performance",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_512x512": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_512x512.safetensors": "BiRefNet_512x512.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Optimized for 512x512 resolution, faster processing",
        "default_res": 512,
        "max_res": 1024,
        "min_res": 256,
        "force_res": True
    },
    "BiRefNet-HR": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-HR.safetensors": "BiRefNet-HR.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "High resolution general purpose model",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet-portrait": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-portrait.safetensors": "BiRefNet-portrait.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Optimized for portrait/human matting",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-matting.safetensors": "BiRefNet-matting.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "General purpose matting model",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet-HR-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-HR-matting.safetensors": "BiRefNet-HR-matting.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "High resolution matting model",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet_lite": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite.safetensors": "BiRefNet_lite.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Lightweight version for faster processing",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_lite-2K": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite-2K.safetensors": "BiRefNet_lite-2K.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Lightweight version optimized for 2K resolution",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet_dynamic": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_dynamic.safetensors": "BiRefNet_dynamic.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Dynamic model for high-resolution dichotomous image segmentation",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_lite-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite-matting.safetensors": "BiRefNet_lite-matting.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Lightweight matting model for general purpose",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_toonout": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_toonout.safetensors": "BiRefNet_toonout.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "A model to get a toon style outline from an image.",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    }
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def handle_model_error(message):
    print(f"[BiRefNetBatched ERROR] {message}")
    raise RuntimeError(message)

def snap32(v: int) -> int:
    """Snap a dimension to the nearest multiple of 32 (>=32).

    BiRefNet's Swin backbone downsamples 32x; non-multiple-of-32 inputs error or get
    silently padded, misaligning the mask. We snap explicitly and warn so the user
    knows their value changed.
    """
    s = max(32, int(round(v / 32) * 32))
    if s != v:
        print(f"[BiRefNetBatched] {v} not divisible by 32, snapped to {s}")
    return s

def refine_foreground(image_bchw, masks_b1hw):
    b, c, h, w = image_bchw.shape
    if b != masks_b1hw.shape[0]:
        raise ValueError("images and masks must have the same batch size")

    image_np = image_bchw.cpu().numpy()
    mask_np = masks_b1hw.cpu().numpy()

    refined_fg = []
    for i in range(b):
        mask = mask_np[i, 0]
        thresh = 0.45
        mask_binary = (mask > thresh).astype(np.float32)

        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        transition_mask = np.logical_and(mask > 0.05, mask < 0.95)

        alpha = 0.85
        mask_refined = np.where(transition_mask,
                              alpha * mask + (1-alpha) * edge_blur,
                              mask_binary)

        edge_region = np.logical_and(mask > 0.2, mask < 0.8)
        mask_refined = np.where(edge_region,
                              mask_refined * 0.98,
                              mask_refined)

        result = []
        for c in range(image_np.shape[1]):
            channel = image_np[i, c]
            refined = channel * mask_refined
            result.append(refined)

        refined_fg.append(np.stack(result))

    return torch.from_numpy(np.stack(refined_fg))

# ---------------------------------------------------------------------------
# Model loading, use_local, and cache  (Task 1)
# ---------------------------------------------------------------------------
# Each variant lives under ComfyUI/models/RMBG/<cache_dir>/ (cache_dir == "BiRefNet"
# for every variant, so the python/config/weights files coexist in one directory).

def _required_files(model_name):
    """Return (weights.safetensors, model .py, BiRefNet_config.py) for this variant.

    The model python file is variant-specific: most variants ship `birefnet.py`, the
    lite variants ship `birefnet_lite.py`. Derive it from MODEL_CONFIG rather than
    hardcoding so every variant works.
    """
    files = list(MODEL_CONFIG[model_name]["files"].keys())
    weights = next(f for f in files if f.endswith(".safetensors"))
    model_py = next(f for f in files if f.endswith(".py") and f != "BiRefNet_config.py")
    return weights, model_py, "BiRefNet_config.py"

def _download_model(model_name, base):
    """Download every file in MODEL_CONFIG for this variant into `base` (HF cache)."""
    os.makedirs(base, exist_ok=True)
    print(f"[BiRefNetBatched] Downloading {model_name} model files...")
    for filename in MODEL_CONFIG[model_name]["files"].keys():
        if os.path.isfile(os.path.join(base, filename)):
            continue
        print(f"[BiRefNetBatched] Downloading {filename}...")
        hf_hub_download(
            repo_id=MODEL_CONFIG[model_name]["repo_id"],
            filename=filename,
            local_dir=base,
        )
    return base

def resolve_model_dir(model_name, use_local):
    """Resolve (and if needed download) the directory holding this variant's files.

    use_local=True  -> never touch the network; error clearly if files are missing.
    use_local=False -> download from HuggingFace only when something is missing.
    """
    subfolder = MODEL_CONFIG[model_name].get("cache_dir", "BiRefNet")
    base = os.path.join(folder_paths.models_dir, "RMBG", subfolder)
    weights, model_py, cfg_py = _required_files(model_name)
    required = (weights, model_py, cfg_py)

    if use_local:
        have_all = all(os.path.isfile(os.path.join(base, f)) for f in required)
        if not have_all:
            raise FileNotFoundError(
                f"[use_local] '{model_name}' not found in {base}. "
                f"Place {weights}, {model_py} and {cfg_py} there, "
                f"or disable use_local to download from HuggingFace."
            )
        return base

    # Download behaviour: only hit the network if a required file is missing.
    have_all = all(os.path.isfile(os.path.join(base, f)) for f in required)
    if not have_all:
        _download_model(model_name, base)
    return base

def load_birefnet_module(model_dir, model_py_name):
    """exec the remote model code with the upstream fix-ups + issue #179 fix.

    Returns (model_module, config_module).
    """
    cfg_path = os.path.join(model_dir, "BiRefNet_config.py")
    model_py = os.path.join(model_dir, model_py_name)

    # Load the config module first; the model code imports BiRefNetConfig from it.
    spec = importlib.util.spec_from_file_location("BiRefNet_config", cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    sys.modules["BiRefNet_config"] = cfg_mod
    spec.loader.exec_module(cfg_mod)

    # Rewrite the relative import IN MEMORY (do not mutate the downloaded/local file).
    # Idempotent: a no-op if the file was already rewritten by the stock node.
    with open(model_py, "r", encoding="utf-8") as f:
        src = f.read().replace("from .BiRefNet_config", "from BiRefNet_config")

    mod_name = f"birefnet_model_{abs(hash(model_py))}"
    mod = types.ModuleType(mod_name)
    mod.__file__ = model_py                       # issue #179: must be set before exec
    mod.__path__ = [os.path.dirname(model_py)]    # issue #179
    sys.modules[mod_name] = mod
    exec(compile(src, model_py, "exec"), mod.__dict__)
    return mod, cfg_mod

def _instantiate_and_load(mod, cfg_mod, model_dir, weights_name):
    """Build the model from the exec'd module and load its safetensors weights."""
    model = mod.BiRefNet(cfg_mod.BiRefNetConfig())
    state_dict = load_file(os.path.join(model_dir, weights_name))
    model.load_state_dict(state_dict)
    return model

_MODEL_CACHE = {}

def get_birefnet_model(model_name, device, dtype, use_local=False, channels_last=False):
    """Load (or fetch from cache) a BiRefNet model ready for inference.

    Cached by everything that changes the loaded object, so multiple checkpoints /
    precisions / layouts can coexist in VRAM on a big GPU with no extra work.
    """
    key = (model_name, str(device), str(dtype), bool(channels_last))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        weights, model_py, _cfg_py = _required_files(model_name)
        model_dir = resolve_model_dir(model_name, use_local)
        mod, cfg_mod = load_birefnet_module(model_dir, model_py)
        model = _instantiate_and_load(mod, cfg_mod, model_dir, weights)
    except FileNotFoundError:
        raise
    except Exception as e:
        handle_model_error(f"Error loading BiRefNet model '{model_name}': {str(e)}")

    model.eval().to(device)
    if dtype == torch.float16:
        model = model.half()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    _MODEL_CACHE[key] = model
    return model

# ---------------------------------------------------------------------------
# Batched GPU inference core  (Task 2)
# ---------------------------------------------------------------------------

# Fraction of the measured VRAM budget the auto-batcher is allowed to plan for.
# The caching allocator fragments, so planning for 100% of "free" reliably OOMs.
_AUTO_BATCH_SAFETY = 0.90

def _is_cuda_device(device):
    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return False

def _is_oom(exc):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

def _free_vram_bytes(device):
    """Bytes this process can still allocate on `device`: device-free + torch's
    cached-but-unused pool. Prefer ComfyUI's accounting so we agree with the rest
    of the graph; fall back to torch's own counters standalone."""
    try:
        from comfy.model_management import get_free_memory
        return int(get_free_memory(device))
    except Exception:
        free, _total = torch.cuda.mem_get_info(device)
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return int(free + cached)

def _plan_auto_chunk(calib, device, headroom_bytes, out_reserve_bytes, remaining):
    """Turn calibration samples [(n_frames, peak_bytes), ...] into a chunk size.

    Linear model: peak(n) = fixed + n * per_frame. With two samples of different n
    we fit both; with one (or two equal-n) samples we take the whole peak as
    per-frame cost (fixed = 0), which over-estimates and is therefore safe.
    """
    (n1, p1), (n2, p2) = calib[0], calib[-1]
    if n2 > n1:
        per_frame = (p2 - p1) / (n2 - n1)
        # Guard against allocator reuse making the delta implausibly small.
        per_frame = max(per_frame, p2 / (2.0 * n2))
        fixed = max(p1 - per_frame * n1, 0.0)
    else:
        per_frame = max(p1, p2) / float(n1)
        fixed = 0.0
    per_frame = max(per_frame, 1.0)

    free = _free_vram_bytes(device)
    budget = (free - headroom_bytes - out_reserve_bytes) * _AUTO_BATCH_SAFETY - fixed
    chunk = int(budget // per_frame)
    chunk = max(1, min(chunk, remaining))

    mb = 1024.0 ** 2
    print(f"[BiRefNetBatched] auto-batch: free {free / mb:.0f} MB, headroom "
          f"{headroom_bytes / mb:.0f} MB, output reserve {out_reserve_bytes / mb:.0f} MB, "
          f"fixed ~{fixed / mb:.0f} MB, ~{per_frame / mb:.0f} MB/frame -> chunk {chunk}")
    if budget < per_frame:
        print("[BiRefNetBatched] auto-batch: budget below one frame's cost; running with "
              "chunk 1 and relying on OOM fallback. Lower vram_headroom_gb or the "
              "processing resolution if this OOMs.")
    return chunk

def _run_chunk(model, xb, device, input_size, out_hw, mean, std, dtype, channels_last):
    """One chunk: upload -> resize -> normalise -> model -> sigmoid -> resize back.
    xb: [b,3,H,W] view (any device). Returns [b,H,W] float32 on `device`."""
    xb = xb.to(device, dtype=dtype, non_blocking=True)
    # input_size is (H, W) -- F.interpolate takes (H, W), never (W, H).
    xb = F.interpolate(xb, size=input_size, mode="bilinear", align_corners=False)
    xb = (xb - mean) / std
    if channels_last:
        # interpolate can drop the channels_last layout; restore it so the convs
        # actually see channels_last input.
        xb = xb.to(memory_format=torch.channels_last)

    pred = model(xb)
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    if isinstance(pred, dict):
        pred = pred.get("pred", next(iter(pred.values())))
    pred = torch.sigmoid(pred).float()
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)                              # [b,1,h,w]

    # Resize the mask back to the ORIGINAL input resolution (H, W).
    pred = F.interpolate(pred, size=out_hw, mode="bilinear", align_corners=False)
    return pred[:, 0]                                         # [b,H,W]

@torch.inference_mode()
def birefnet_batch_infer(model, images_bhwc, device, input_size,
                         batch_chunk=16, dtype=torch.float16, channels_last=False,
                         auto_batch=False, vram_headroom_gb=1.0, out_device=None):
    """
    images_bhwc : ComfyUI IMAGE tensor [B,H,W,C], float32 in 0..1
    input_size  : (H, W) processing resolution, both multiples of 32
    returns     : masks [B,H,W] float32 in 0..1, resized back to original H,W

    No PIL, no autocast, no per-chunk device->host sync. Inputs are uploaded to
    the device per chunk (not the whole batch up front) to bound peak VRAM; when
    the input already lives on `device` the per-chunk .to() is a no-op view.

    Manual mode (auto_batch=False): fixed chunks of `batch_chunk`; masks stay on
    `device`. Auto mode (CUDA only): the first two chunks (1 and 2 frames -- their
    masks are kept, nothing is thrown away) are profiled with the CUDA allocator's
    peak counters, a per-frame cost is fitted, and every following chunk is sized
    to fill free VRAM minus `vram_headroom_gb`. Masks accumulate on `out_device`
    (the input's device) so a long video doesn't pile its outputs up in VRAM.
    Either mode: an OOM inside a chunk halves the chunk and retries the same frames.
    """
    B, H, W, C = images_bhwc.shape
    assert C == 3, images_bhwc.shape

    x = images_bhwc.permute(0, 3, 1, 2)                     # [B,C,H,W] view

    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device, dtype=dtype).view(1, 3, 1, 1)

    use_auto = bool(auto_batch)
    if use_auto and not _is_cuda_device(device):
        print(f"[BiRefNetBatched] auto_batch needs a CUDA device (got {device}); "
              f"falling back to batch_chunk={batch_chunk}")
        use_auto = False
    if out_device is None or not use_auto:
        out_device = device
    out_device = torch.device(out_device)
    # Outputs that stay on the inference device eat into the budget for later chunks.
    out_bytes_per_frame = H * W * 4 if out_device.type == "cuda" else 0

    headroom_bytes = int(max(0.0, float(vram_headroom_gb)) * 1024 ** 3)
    calib = []                 # [(n_frames, peak_bytes)] for the auto planner
    planned = not use_auto     # auto: chunk size still to be decided
    chunk = 1 if use_auto else max(1, int(batch_chunk))

    out = []
    s = 0
    while s < B:
        n = min(chunk, B - s)
        if use_auto:
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
        try:
            pred = _run_chunk(model, x[s:s + n], device, input_size, (H, W),
                              mean, std, dtype, channels_last)
        except Exception as e:
            if not _is_oom(e) or n == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk = max(1, n // 2)
            print(f"[BiRefNetBatched] OOM at chunk {n}; retrying frames {s}.. with chunk {chunk}")
            continue

        out.append(pred.to(out_device, non_blocking=True))
        del pred
        s += n

        if use_auto and not planned:
            calib.append((n, torch.cuda.max_memory_allocated(device) - base))
            if len(calib) == 1 and s < B:
                chunk = 2          # second calibration point; 1 frame if only 1 left
            elif s < B:
                remaining = B - s
                chunk = _plan_auto_chunk(calib, device, headroom_bytes,
                                         remaining * out_bytes_per_frame, remaining)
                planned = True

    # The non_blocking device->host copies above land in pinned host buffers
    # asynchronously; the host must wait for them before reading (torch.cat).
    # Earlier chunks get flushed incidentally by the next chunk's upload, but the
    # last one doesn't -- without this its masks come out as zeros.
    if _is_cuda_device(device) and out_device.type == "cpu":
        torch.cuda.synchronize(device)

    return torch.cat(out, dim=0)                              # [B,H,W] on out_device

# ---------------------------------------------------------------------------
# Single readback, then mask blur / offset / invert in PIL  (Task 3)
# ---------------------------------------------------------------------------

def postprocess_masks_pil(masks_bhw, mask_blur, mask_offset, invert_output):
    """Apply the stock node's edge feathering to the one-channel output masks.

    masks_bhw : float tensor [B,H,W] in 0..1, already on CPU (after the single readback).
    Mirrors the stock node exactly: GaussianBlur -> Max/MinFilter offset -> invert, each
    on a mode 'L' uint8 image. In Pillow GaussianBlur(radius=r) uses r directly as the
    gaussian std-dev, so mask_blur is passed straight through (no calibration).
    Returns a float tensor [B,H,W] in 0..1 (still on CPU).
    """
    out = []
    for m in masks_bhw:
        img = Image.fromarray((m.numpy() * 255.0).astype(np.uint8), mode="L")
        if mask_blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=mask_blur))
        if mask_offset > 0:
            for _ in range(mask_offset):
                img = img.filter(ImageFilter.MaxFilter(3))
        elif mask_offset < 0:
            for _ in range(-mask_offset):
                img = img.filter(ImageFilter.MinFilter(3))
        if invert_output:
            img = Image.fromarray(255 - np.array(img))
        out.append(torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0))
    return torch.stack(out, dim=0)                            # [B,H,W]

def _hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a = 255
    elif len(hex_color) == 8:
        r, g, b, a = (int(hex_color[0:2], 16), int(hex_color[2:4], 16),
                      int(hex_color[4:6], 16), int(hex_color[6:8], 16))
    else:
        raise ValueError("Invalid color format")
    return (r, g, b, a)

# ---------------------------------------------------------------------------
# Node interface  (Tasks 4 & 5)
# ---------------------------------------------------------------------------

class BiRefNetRMBGBatched:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image(s) to be processed for background removal. A batch is run through the GPU in chunks.",
            "model": "Select the BiRefNet model variant to use.",
            "width": "Processing width. Snapped to a multiple of 32 in code (Swin backbone requirement). Pin this per resolution/orientation path.",
            "height": "Processing height. Snapped to a multiple of 32 in code (Swin backbone requirement). Pin this per resolution/orientation path.",
            "batch": "Batch processing. ON: push the whole input through the GPU in chunks of batch_chunk (fast for many frames). OFF: process one image at a time — for single images / when you don't need batching; batch_chunk is ignored. width/height still apply either way.",
            "batch_chunk": "How many images to push through the GPU at once when batch is ON and auto_batch is OFF. Bounds peak activation memory; benchmark 8/16/24/32/49 on your hardware. Ignored when auto_batch is ON.",
            "auto_batch": "VRAM-optimized batching (CUDA only). Profiles the first 1- and 2-frame chunks with the CUDA allocator, fits a per-frame cost, then sizes every following chunk to fill free VRAM minus vram_headroom_gb. Nothing is thrown away: the profiling chunks' masks are real output. Works with either precision and with/without channels_last. Any OOM halves the chunk and retries. Falls back to batch_chunk on non-CUDA devices.",
            "vram_headroom_gb": "VRAM (GB) auto_batch must leave free for the rest of the workflow / other nodes. Raise it if downstream nodes OOM; lower it (even 0) on a GPU dedicated to this node. Only used when auto_batch is ON.",
            "precision": "fp16 (model + input cast to float16, no autocast) or fp32. fp16 is the fast path on CUDA.",
            "channels_last": "Use channels_last memory format on model and input. May speed up some variants on Tensor Cores; benchmark on/off.",
            "use_local": "Load the model from ComfyUI/models/RMBG/BiRefNet/ only, never downloading. Errors clearly if files are missing.",
            "mask_blur": "Amount of blur to apply to the mask edges (0 = none). In Pillow this value is the gaussian std-dev.",
            "mask_offset": "Adjust the mask boundary (positive expands, negative shrinks).",
            "invert_output": "Invert the output mask.",
            "refine_foreground": "Use Fast Foreground Colour Estimation to optimize transparent background.",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Choose background color (Alpha = transparent).",
            "unload_after_run": "Drop all cached BiRefNet models from VRAM after this run. The model cache is process-global and otherwise keeps every model/precision/layout combo resident forever. Next run reloads (slower), but this bounds VRAM on smaller GPUs.",
        }
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (list(MODEL_CONFIG.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32, "tooltip": tooltips["width"]}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32, "tooltip": tooltips["height"]}),
                "batch": ("BOOLEAN", {"default": True, "tooltip": tooltips["batch"]}),
                "batch_chunk": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1, "tooltip": tooltips["batch_chunk"]}),
                "auto_batch": ("BOOLEAN", {"default": False, "tooltip": tooltips["auto_batch"]}),
                "vram_headroom_gb": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.25, "tooltip": tooltips["vram_headroom_gb"]}),
                "precision": (["fp16", "fp32"], {"default": "fp16", "tooltip": tooltips["precision"]}),
                "channels_last": ("BOOLEAN", {"default": False, "tooltip": tooltips["channels_last"]}),
                "use_local": ("BOOLEAN", {"default": False, "tooltip": tooltips["use_local"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "refine_foreground": ("BOOLEAN", {"default": False, "tooltip": tooltips["refine_foreground"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
                "unload_after_run": ("BOOLEAN", {"default": False, "tooltip": tooltips["unload_after_run"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "🧪AILab/🧽RMBG"

    def process_image(self, image, model, width=1024, height=1024, batch_chunk=16,
                      batch=True, auto_batch=False, vram_headroom_gb=1.0,
                      precision="fp16", channels_last=False, use_local=False,
                      mask_blur=0, mask_offset=0, invert_output=False,
                      refine_foreground=False, background="Alpha",
                      background_color="#222222", unload_after_run=False):
        try:
            # 0. Accept RGBA input (e.g. this node's own IMAGE output chained back in):
            #    drop the alpha channel. The stock node did PIL convert("RGB").
            if image.shape[-1] == 4:
                image = image[..., :3]
            # 1. Snap processing dims to /32 (Swin backbone requirement).
            width, height = snap32(width), snap32(height)
            # 2. Precision -> dtype.
            dtype = torch.float16 if precision == "fp16" else torch.float32
            # batch OFF -> process one image at a time (chunk size 1); batch_chunk and
            # auto_batch ignored. width/height still apply. For a single image this is
            # identical to batch ON. batch ON + auto_batch -> chunk sized from free VRAM.
            use_auto = bool(batch and auto_batch)
            effective_chunk = batch_chunk if batch else 1
            if not batch:
                mode = "single (batch off)"
            elif use_auto:
                mode = f"auto-batch (headroom {vram_headroom_gb:g} GB)"
            else:
                mode = f"chunk={batch_chunk}"
            print(f"[BiRefNetBatched] {model} | {image.shape[0]} frame(s) @ {width}x{height} "
                  f"(WxH) | {precision} | {mode} | channels_last={channels_last}")

            # 3. Load (or fetch cached) model.
            net = get_birefnet_model(model, device, dtype, use_local, channels_last)

            # 4. Batched GPU inference. input_size is (H, W). Result is [B,H,W], already
            #    resized back to the original frame resolution -- on the inference device
            #    (manual mode) or on the input image's device (auto_batch mode).
            masks = birefnet_batch_infer(
                net, image, device, (height, width),
                batch_chunk=effective_chunk, dtype=dtype, channels_last=channels_last,
                auto_batch=use_auto, vram_headroom_gb=vram_headroom_gb,
                out_device=image.device,
            )

            # 5. Edge feathering. Only touches the host when actually requested: with
            #    blur=0 / offset=0 / no invert the PIL step is skipped entirely, so the
            #    masks keep full float precision (stock always round-trips through
            #    uint8) and GPU-resident inputs see ZERO device->host transfers.
            masks = masks.clamp(0, 1)
            if mask_blur > 0 or mask_offset != 0 or invert_output:
                # The single device->host transfer: PIL feathering on the one-channel
                # output mask, bit-parity with the stock node.
                masks = postprocess_masks_pil(masks.cpu(), mask_blur, mask_offset, invert_output)

            # 6. Composite on the input image's own device (GPU for VAE-decoded batches,
            #    CPU for LoadImage) -- the input is never forced through a host round-trip.
            out_device = image.device
            masks = masks.to(out_device)
            image = image.float()                             # [B,H,W,3] in 0..1
            alpha = masks.unsqueeze(-1)                        # [B,H,W,1]

            if refine_foreground:
                # numpy/cv2 path internally (CPU); result moved back to out_device.
                rgb = refine_foreground_batch(image, masks).to(out_device)   # [B,H,W,3]
            else:
                rgb = image

            if background == "Alpha":
                images_out = torch.cat([rgb, alpha], dim=-1)  # [B,H,W,4] RGBA
            else:
                r, g, b, _a = _hex_to_rgba(background_color)
                bg = torch.tensor([r, g, b], dtype=torch.float32,
                                  device=out_device).view(1, 1, 1, 3) / 255.0
                # NOTE: with refine_foreground the refined rgb is already mask-weighted,
                # so the mask is multiplied in twice here. The stock node does exactly
                # the same (PIL alpha_composite over an already-premultiplied
                # foreground) -- kept as-is for output parity with stock.
                images_out = rgb * alpha + bg * (1.0 - alpha)  # [B,H,W,3] composited

            mask_image_out = masks.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

            if unload_after_run:
                # Drop every cached model so ComfyUI / other packs can reclaim VRAM.
                del net
                _MODEL_CACHE.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return (images_out.contiguous(), masks, mask_image_out)
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")


def refine_foreground_batch(image_bhwc, masks_bhw):
    """Batched wrapper around refine_foreground; returns [B,H,W,3] in 0..1.

    Accepts CPU or GPU tensors; the actual estimation runs in numpy/cv2 on CPU
    (refine_foreground calls .cpu() internally), so the result comes back on CPU.
    """
    img_bchw = image_bhwc.permute(0, 3, 1, 2).contiguous()    # [B,3,H,W]
    refined = refine_foreground(img_bchw, masks_bhw.unsqueeze(1))  # [B,3,H,W]
    return refined.permute(0, 2, 3, 1).clamp(0, 1)            # [B,H,W,3]


# Node Mapping
NODE_CLASS_MAPPINGS = {
    "BiRefNetRMBG_Batched": BiRefNetRMBGBatched
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNetRMBG_Batched": "BiRefNet Batched (RMBG)"
}
