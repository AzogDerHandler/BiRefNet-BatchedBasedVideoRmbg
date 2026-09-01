# BiRefNet-BatchedBasedVideoRmbg

I realized that ai video matting models kinda suck and lack behind the precision of BiRefNet when it comes to video rmbg. The issue is that classic implementation of image based rmbg takes forever on a video clip, since it not utilizes all compute resources if you use a high end gpu (h100, A100 etc.). Thats why i built this batch node, it uses up all compute available and gets the job done significantly faster. 
I build this as a standalone ComfyUI node pack containing a single node — **BiRefNet Batched (RMBG)**
(class key `BiRefNetRMBG_Batched`) — BiRefNet background removal with a true batched
GPU pipeline.

Derived from the BiRefNet node of [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)
(GPL-3.0, see `LICENSE`), replacing its per-image, PIL-based inference loop with
chunked batched tensor inference. It is **not** a fork of the pack: it ships only this
one node, registers its own class key, and can be installed **alongside** the original
ComfyUI-RMBG — both nodes will appear in the `🧪AILab/🧽RMBG` category and workflows
can mix them freely. 

## Install

Copy this folder into `ComfyUI/custom_nodes/` and install the dependencies:

```bash
pip install -r requirements.txt
```

## What the node does differently from the stock BiRefNet node

- **Batched GPU inference.** Input batch → resize → ImageNet-normalize → model →
  sigmoid → resize-back all stay on the GPU, in chunks of `batch_chunk` images, with a
  single device→host transfer per batch. The stock node loops one PIL image at a time.
- **Explicit `width`/`height`** processing resolution (snapped to multiples of 32, the
  Swin backbone requirement). The stock node derives resolution from the model variant;
  here it defaults to 1024×1024 — set it to the model's native size (e.g. 2048×2048 for
  BiRefNet-HR) to reproduce stock quality.
- **`precision`** (`fp16` default / `fp32`) via model+input casting, no autocast.
- **`channels_last`** memory-format toggle.
- **`use_local`**: load strictly from `ComfyUI/models/RMBG/BiRefNet/`, never touching
  the network, with a clear error listing missing files.
- **`batch` toggle**: OFF processes one image at a time (`batch_chunk` ignored).
- **`auto_batch` — VRAM-optimized batching** (CUDA only). Instead of a fixed
  `batch_chunk`, the node fills whatever VRAM is free: it runs the first 1-frame and
  2-frame chunks with the CUDA allocator's peak counters on (their masks are kept, so no
  work is wasted), fits `peak = fixed + n × per_frame`, then sizes every later chunk to
  `(free VRAM − vram_headroom_gb − space for the remaining output masks) × 0.9`. Pick
  `precision` (fp16/fp32) and `channels_last` as usual — the measurement is taken with
  exactly the combination you chose, so the chunk size adapts to it. Masks accumulate on
  the input image's device so a long video doesn't stack its outputs in VRAM. A summary
  line (`free / headroom / MB per frame -> chunk N`) is printed to the console.
- **`vram_headroom_gb`**: VRAM `auto_batch` leaves untouched for the rest of the
  workflow (default 1 GB). Raise it if downstream nodes OOM; set 0 on a dedicated GPU.
- **OOM fallback** (both modes): an out-of-memory error inside a chunk halves the chunk
  and retries the same frames, instead of failing the whole run.
- **`unload_after_run`**: drop all cached models from VRAM after the run (the cache
  otherwise keeps every model/precision/layout combo resident forever) — for smaller GPUs.
- Mask feathering (`mask_blur`, `mask_offset`, `invert_output`) runs in PIL on the
  one-channel output mask after the single readback — bit-for-bit identical to the
  stock node's feathering. When all three are off the PIL step is skipped entirely,
  keeping full float mask precision (better than stock, which always quantizes to uint8).
- Inputs upload to the GPU **per chunk** (bounded peak VRAM), compositing happens on the
  input image's own device (GPU batches never round-trip through the host), and the node
  follows ComfyUI's device selection (`--cpu`, `--cuda-device`, MPS) rather than
  hardcoding `"cuda"`.

All BiRefNet variants are supported (general, HR, lite, matting, portrait, toonout,
512×512, …) with no per-checkpoint special-casing. Model weights: Apache-2.0,
by [ZhengPeng7](https://huggingface.co/ZhengPeng7).
