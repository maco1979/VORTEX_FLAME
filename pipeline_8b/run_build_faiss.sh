#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
pip install faiss-cpu sentence-transformers -q 2>/dev/null
python /mnt/d/VORTEX_FLAME/pipeline_8b/build_cs_faiss.py
