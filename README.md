# 🐝 HYVE - Hybrid Vertex Encoder for Neural Distance Fields

HYVE is a hybrid neural implicit representation that encodes 3D geometry into a latent grid via a hybrid graph and grid neural network encoder and decodes it at specified positions into a signed distance field (SDF). It is trained fully unsupervised (using the Eikonal equation) and is able to deal well with varying point densities, point counts and point connectivity. If you have any trouble or want to ask a question, feel free to reach out by [mail](mailto:contact@srjeske.de) or to [open an issue](https://github.com/InteractiveComputerGraphics/hyve/issues/new).

## 📢 News

- **[2026/02/06]** 🎉 Initial code release.
- **[2026/01/28]** 📝 Paper accepted at [IEEE TVCG](https://ieeexplore.ieee.org/document/11367343)!

## 🔗 Links

- [Paper PDF](https://srjeske.de/publications/2026-tvcg-hyve/JKMB26.pdf)
- [Paper Website](hyve.physics-simulation.org)
- [Supplemental Video (YouTube)](https://youtu.be/ykKsjeX2-0M)
- [Datasets (HuggingFace Hub)](https://huggingface.co/datasets/stefan-jeske/hyve-datasets)

## 🛠️ Installation/Setup

Clone the repo and use UV to install all dependencies.

```sh
git clone https://github.com/InteractiveComputerGraphics/hyve.git
cd hyve
uv sync
source .venv/bin/activate
```

## 💾 Downloading Datasets

The datasets are hosted in a separate repository on HuggingFace Hub.

```shell
# Download all datasets (51.81GB)
hf download stefan-jeske/hyve-datasets --repo-type dataset --local-dir datasets/
```

Or download specific subsets:

- **🍯 Honey (55.6MB):** `hf download stefan-jeske/hyve-datasets --include honey/* --repo-type dataset --local-dir datasets/`
- **🛡️ Armadillo (17.49GB):** `hf download stefan-jeske/hyve-datasets --include armadillo/* --repo-type dataset --local-dir datasets/`
- **🐉 Dragon (9.31GB):** `hf download stefan-jeske/hyve-datasets --include dragon/* --repo-type dataset --local-dir datasets/`
- **🏺 Objaverse (18.11GB):** `hf download stefan-jeske/hyve-datasets --include objaverse/* --repo-type dataset --local-dir datasets/`
- **🗿 Thingi10k (6.8GB):** `hf download stefan-jeske/hyve-datasets --include thingi10k/* --repo-type dataset --local-dir datasets_test/`

The other datasets used in the paper, namely **ShapeNet V2**, **Dynamic FAUST** or **ScanNet** could not be hosted due to license restrictions.
However, we provide the scripts we used to generate the datasets from the raw data of each dataset in the scripts folder, as well as any config files in the respective dataset folder.
The code for loading these datasets is contained in `hyve/dataset/` and for pre-processing in `hyve/scripts`. However, no checkpoints or model configs are provided, because it's much more difficult to describe a universally reproducible workflow.

## 🚀 Usage

### Quick Start (Honeycomb/Teaser Reconstruction)

To run a reconstruction on the honeycomb example using a pre-trained model:

```sh
./run_honeycomb.sh
```

The reconstructions will be saved in `experiments/honey/*/reconstructions`.

*Note: "Honeycomb" (https://skfb.ly/onPMp) by RISD Nature Lab is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).*

You can also generate all the examples from the teaser in the paper by running the following script, which will download the required models from HuggingFace and will subsequently generate models in `experiments/honey/*/reconstructions` and `experiments/objaverse/*/reconstructions`.

```sh
./run_teaser.sh
```

### Training and Testing

The project uses PyTorch Lightning CLI. You can train or test the model using `hyve/train.py`.

Example: Inference (test only) on the Honey dataset using an Objaverse-trained model:

```sh
python hyve/train.py \
    --config models/objaverse/pic-normals.yaml \
    --config datasets/honey/honey.yaml \
    --ckpt_path models/objaverse/pic-normals.ckpt \
    --trainer.logger.save_dir experiments/honey/ \
    --trainer.logger.version=honey \
    --test_only
```

Example: Training on the dragon dataset:

```sh
python hyve/train.py --config models/dragon/pic-basic.yaml
# Or again using uv
uv run hyve/train.py --config models/dragon/pic-basic.yaml
```

### Minimal Example Script

A minimal python script `minimal_example.py` is included to demonstrate how to load a model and perform inference on a custom mesh (e.g. the honeycomb data).

```sh
python minimal_example.py
# Or if using UV
uv run minimal_example.py
```

## 📦 Pre-trained Models

Checkpoints and configurations are provided in the `models/` directory for different datasets:

- `models/armadillo/`
- `models/dragon/`
- `models/objaverse/`
- `models/thingi10k/`

Each folder typically contains variants like `basic.ckpt`, `normals.ckpt`, `pic-basic.ckpt`, and `pic-normals.ckpt`.
Models for the other datasets used in the paper, namely **ShapeNet V2**, **Dynamic FAUST** or **ScanNet** may be added in the future, particularly if there is specific demand.

## 📝 Citation

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

## 💻 Requirements

- Linux or Windows
- CUDA-capable GPU (at least ~8GB VRAM)
- CUDA Toolkit (`nvcc`)
- C++ Compiler (`gcc`)

Verfiy your compiler setup:
```shell
nvcc --version
gcc --version
```