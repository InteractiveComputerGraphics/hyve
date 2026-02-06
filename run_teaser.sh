#!/bin/bash
uv sync
source .venv/bin/activate

# Download beehive
hf download stefan-jeske/hyve-datasets --include honey/* --repo-type dataset --local-dir datasets/

# Download necessary files for objaverse
hf download stefan-jeske/hyve-datasets \
  objaverse/train.json \
  objaverse/test_teaser.json \
  objaverse/processed/20_test_2_200000.pt \
  objaverse/processed/29_test_2_200000.pt \
  objaverse/processed/48_test_2_200000.pt \
  objaverse/processed/55_test_2_200000.pt \
   --repo-type dataset --local-dir datasets/


# Beehive/Honeycomb
python hyve/train.py --config models/objaverse/pic-normals.yaml  --config datasets/honey/honey.yaml --ckpt_path models/objaverse/pic-normals.ckpt --trainer.logger.save_dir experiments/honey/ --trainer.logger.version=honey --test_only
# Select objaverse models
python hyve/train.py --config models/objaverse/pic-normals.yaml --ckpt_path models/objaverse/pic-normals.ckpt --data.test_split=datasets/objaverse/test_teaser.json --test_only --data.save_input_pointcloud=experiments/objaverse/input_points/
