#!/bin/bash
uv sync
source .venv/bin/activate

# Dragon models
python hyve/train.py --config models/dragon/basic.yaml          --ckpt_path models/dragon/basic.ckpt          --trainer.limit_test_batches=2
python hyve/train.py --config models/dragon/normals.yaml        --ckpt_path models/dragon/normals.ckpt        --trainer.limit_test_batches=2
python hyve/train.py --config models/dragon/pic-basic.yaml      --ckpt_path models/dragon/pic-basic.ckpt      --trainer.limit_test_batches=2
python hyve/train.py --config models/dragon/pic-normals.yaml    --ckpt_path models/dragon/pic-normals.ckpt    --trainer.limit_test_batches=2

# Armadillo models
python hyve/train.py --config models/armadillo/basic.yaml       --ckpt_path models/armadillo/basic.ckpt       --trainer.limit_test_batches=2
python hyve/train.py --config models/armadillo/normals.yaml     --ckpt_path models/armadillo/normals.ckpt     --trainer.limit_test_batches=2
python hyve/train.py --config models/armadillo/pic-basic.yaml   --ckpt_path models/armadillo/pic-basic.ckpt   --trainer.limit_test_batches=2
python hyve/train.py --config models/armadillo/pic-normals.yaml --ckpt_path models/armadillo/pic-normals.ckpt --trainer.limit_test_batches=2

# Thingi10k models
python hyve/train.py --config models/thingi10k/basic.yaml       --ckpt_path models/thingi10k/basic.ckpt       --trainer.limit_test_batches=2
python hyve/train.py --config models/thingi10k/normals.yaml     --ckpt_path models/thingi10k/normals.ckpt     --trainer.limit_test_batches=2
python hyve/train.py --config models/thingi10k/pic-basic.yaml   --ckpt_path models/thingi10k/pic-basic.ckpt   --trainer.limit_test_batches=2
python hyve/train.py --config models/thingi10k/pic-normals.yaml --ckpt_path models/thingi10k/pic-normals.ckpt --trainer.limit_test_batches=2

# Objaverse models
python hyve/train.py --config models/objaverse/basic.yaml       --ckpt_path models/objaverse/basic.ckpt       --trainer.limit_test_batches=2
python hyve/train.py --config models/objaverse/normals.yaml     --ckpt_path models/objaverse/normals.ckpt     --trainer.limit_test_batches=2
python hyve/train.py --config models/objaverse/pic-basic.yaml   --ckpt_path models/objaverse/pic-basic.ckpt   --trainer.limit_test_batches=2
python hyve/train.py --config models/objaverse/pic-normals.yaml --ckpt_path models/objaverse/pic-normals.ckpt --trainer.limit_test_batches=2