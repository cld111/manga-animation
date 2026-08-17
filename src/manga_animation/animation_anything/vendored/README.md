# Vendored AnimateAnything model code

This directory contains the inference subset of the upstream
[alibaba/animate-anything](https://github.com/alibaba/animate-anything) repository
(commit `82 commits / main` as fetched for this integration), Apache-2.0 licensed (see
`LICENSE`). Files are copied unmodified except for adding the empty `__init__.py` markers so
`models` / `utils` import as regular packages:

- `models/pipeline.py` — `LatentToVideoPipeline` (the mask/motion-conditioned video pipeline).
- `models/unet_3d_condition_mask.py` — custom `UNet3DConditionModel` with mask + motion conditioning.
- `models/unet_3d_blocks.py` — the 3D down/up/mid blocks used by that UNet.
- `utils/common.py` — `tensor_to_vae_latent`, `DDPM_forward_timesteps`, motion-scoring helpers.

This code is imported ONLY by `../worker.py`, which runs in the isolated environment with
AnimateAnything's pinned dependency stack (see its `requirements.txt`: diffusers==0.24.0,
transformers==4.36.2, torch==2.0.0). It must never be imported by the main pipeline process,
whose `ml` stack (transformers>=5.0) is incompatible.

The pretrained checkpoint (`animate_anything_512_v1.02`) is downloaded separately to the
remote worker and passed by path; it is not part of this vendored source tree.
