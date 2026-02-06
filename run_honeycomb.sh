#!/bin/bash

uv sync
source .venv/bin/activate

# Download the necessary models
hf download stefan-jeske/hyve-datasets --include honey/* --repo-type dataset --local-dir datasets/

# Model with feature projection -> good results and single layer surface
python hyve/train.py --config models/objaverse/pic-normals.yaml  --config datasets/honey/honey.yaml --ckpt_path models/objaverse/pic-normals.ckpt --trainer.logger.save_dir experiments/honey/ --trainer.logger.version=honey --test_only

# Model without feature projection -> slightly better metrics but/because of double layer surface
python hyve/train.py --config models/objaverse/normals.yaml  --config datasets/honey/honey.yaml --ckpt_path models/objaverse/normals.ckpt --trainer.logger.version=honey --trainer.logger.save_dir experiments/honey/ --test_only