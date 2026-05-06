#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
echo "Python: $(which python)" > /mnt/d/VORTEX_FLAME/pipeline_8b/diag_output.txt
python -c 'import torch; print("Torch:", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' >> /mnt/d/VORTEX_FLAME/pipeline_8b/diag_output.txt 2>&1
echo "Starting diagnostic..." >> /mnt/d/VORTEX_FLAME/pipeline_8b/diag_output.txt
python /mnt/d/VORTEX_FLAME/pipeline_8b/full_diagnostic_7b.py >> /mnt/d/VORTEX_FLAME/pipeline_8b/diag_output.txt 2>&1
echo "Done. Exit code: $?" >> /mnt/d/VORTEX_FLAME/pipeline_8b/diag_output.txt
