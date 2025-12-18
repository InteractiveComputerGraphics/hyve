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
    { name = "Paper", url = "", icon = "pdf" },
    { name = "arXiv", url = "https://arxiv.org/abs/2310.06644", icon = "ai" },
    { name = "Video", url = "", icon = "youtube" },
    { name = "Code", url = "https://github.com/InteractiveComputerGraphics/hyve", icon = "github" },
]
_teaser_video = ""
+++

# Abstract

Neural shape representation generally refers to representing 3D geometry using neural networks, e.g., computing a signed distance or occupancy value at a specific spatial position. 
In this paper we present a neural-network architecture suitable for accurate encoding of 3D shapes in a single forward pass.
Our architecture is based on a multi-scale hybrid system incorporating graph-based and voxel-based components, as well as a continuously differentiable decoder. 
The hybrid system includes a novel way of voxelizing point-based features in neural networks by projecting the point ``feature-field'' onto a grid.
This projection is insensitive to local point density, and we show that it can be used to obtain smoother and more detailed reconstructions, in particular when combined with oriented point clouds as input.
Our architecture also requires only a single forward pass, instead of the latent-code optimization used in auto-decoder methods.
Furthermore, our network is trained to solve the well-established eikonal equation and only requires knowledge of the zero-level set for training and inference. 
We additionally propose a modification to the aforementioned loss function for the case that surface normals are not well defined, e.g., in the context of non-watertight surfaces and non-manifold geometry.
Overall, our method consistently outperforms other baselines on the surface reconstruction task across a wide variety of datasets, while being more computationally efficient and requiring fewer parameters.

## More coming soon!

# BibTeX

```bibtex
@misc{jeskeHYVEHybridVertex2024,
  title = {{{HYVE}}: {{Hybrid Vertex Encoder}} for {{Neural Distance Fields}}},
  author = {Jeske, Stefan Rhys and Klein, Jonathan and Michels, Dominik L. and Bender, Jan},
  year = 2024,
  month = aug,
  number = {arXiv:2310.06644},
  eprint = {2310.06644},
  primaryclass = {cs},
  doi = {10.48550/arXiv.2310.06644},
}
```
