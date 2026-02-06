+++
title = "HYVE: Hybrid Vertex Encoder for Neural Distance Fields"
description = "Project page for HYVE."

[extra]
authors = [
    { name = "Stefan R. Jeske", url = "https://srjeske.de", affiliation_indices = [0] },
    { name = "Jonathan Klein", url = "https://jonathank.de/", affiliation_indices = [1] },
    { name = "Dominik Michels", url = "https://dmichels.de/", affiliation_indices = [1] },
    { name = "Jan Bender", url = "https://animation.rwth-aachen.de/person/1/", affiliation_indices = [0] },
]
affiliations = [
    { name = "RWTH Aachen University", url = "https://animation.rwth-aachen.de/" },
    { name = "KAUST", url = "https://computationalsciences.org/" },
]
links = [
    { name = "Paper", url = "2026-TVCG-HYVE.pdf", icon = "pdf" },
    { name = "Version of Record", url = "https://dx.doi.org/10.1109/TVCG.2026.3658870", icon = "" },
    { name = "Video", url = "https://www.youtube.com/watch?v=ykKsjeX2-0M", icon = "youtube" },
    { name = "Code", url = "https://github.com/InteractiveComputerGraphics/hyve", icon = "github" },
    { name = "Data", url = "https://huggingface.co/datasets/stefan-jeske/hyve-datasets", icon = "data" },
]
teaser_video = "/video/objaverse_720p.webm"
+++

## :paperclip: Abstract

{{ inline_image(path="representative_image.png", width=600, alt="Representative Image", side="right", caption="Reconstruction of a Beehive using our method.") }}

Neural shape representation generally refers to representing 3D geometry using neural networks, e.g., computing a signed distance or occupancy value at a specific spatial position. 
In this paper we present a neural-network architecture suitable for accurate encoding of 3D shapes in a single forward pass.
Our architecture is based on a multi-scale hybrid system incorporating graph-based and voxel-based components, as well as a continuously differentiable decoder. 
The hybrid system includes a novel way of voxelizing point-based features in neural networks by projecting the point "feature-field" onto a grid.
This projection is insensitive to local point density, and we show that it can be used to obtain smoother and more detailed reconstructions, in particular when combined with oriented point clouds as input.
Our architecture also requires only a single forward pass, instead of the latent-code optimization used in auto-decoder methods.
Furthermore, our network is trained to solve the well-established eikonal equation and only requires knowledge of the zero-level set for training and inference. 
We additionally propose a modification to the aforementioned loss function for the case that surface normals are not well defined, e.g., in the context of non-watertight surfaces and non-manifold geometry.
Overall, our method consistently outperforms other baselines on the surface reconstruction task across a wide variety of datasets, while being more computationally efficient and requiring fewer parameters.

## :rocket: Quickstart

Its very easy to get started with our method simply run the following commands

```sh,data-copy
git clone https://github.com/InteractiveComputerGraphics/hyve.git
cd hyve
uv sync

# Download honeycomb model from HF and run inference using two different models
./run_honeycomb.sh
```

*Requires: Linux or Windows with CUDA*

## :framed_picture: Gallery

### Feature Projection vs Pooling

{{ carousel(paths=["/video/thingi10k_proj.webm", "/video/armadillo_proj.webm", "/video/scannet_proj.webm",  "/video/dragon_proj.webm", "/video/dfaust_proj.webm"]) }}

### Comparison to Related Work

{{ carousel(paths=["/video/thingi10k_comp.webm", "/video/armadillo_comp.webm", "/video/shapenet_comp.webm", "/video/scannet_comp.webm",  "/video/dragon_comp.webm", "/video/dfaust_comp.webm"]) }}

## :page_with_curl: Paper

You can view the paper below or download it via the link at the top.

{{ pdf(path="2026-TVCG-HYVE.pdf") }}

## :scroll: BibTeX

```bibtex
@article{jeskeHYVEHybridVertex2026,
  title = {{{HYVE}}: {{Hybrid Vertex Encoder}} for {{Neural Distance Fields}}},
  shorttitle = {{{HYVE}}},
  author = {Jeske, Stefan R. and Klein, Jonathan and Michels, Dominik and Bender, Jan},
  year = 2026,
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  pages = {1--12},
  doi = {10.1109/TVCG.2026.3658870},
  copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html}
}
```
