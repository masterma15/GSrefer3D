# Third-party components

This repository is a **research integration** project. Most code volume comes from upstream; **original contribution** is primarily `bridge/` and the files listed in [patches/README.md](patches/README.md).

| Component | Upstream | License (check upstream for exact terms) |
|-----------|----------|----------------------------------------|
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Inria GRAPHDECO | Non-commercial research / evaluation — see upstream `LICENSE.md` |
| [RoboRefer](https://github.com/Zhoues/RoboRefer) | Zhoues et al. | See upstream repository |
| [RefSpatial / Expand-Bench](https://huggingface.co/datasets/BAAI/RefSpatial-Bench) | BAAI / dataset authors | Dataset terms on Hugging Face |
| Depth Anything V2 | LiheYoung/Depth-Anything-V2 | See upstream |

**Not redistributed in this Git repo:** model weights (RoboRefer-2B-SFT, LoRA adapters), COLMAP binaries, SIBR viewer binaries, training images, `point_cloud.ply`, or full benchmark image packs. Download separately — [docs/UPSTREAM_SETUP.md](docs/UPSTREAM_SETUP.md).
