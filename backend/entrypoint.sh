#!/bin/sh
set -e

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
