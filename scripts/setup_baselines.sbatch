#!/usr/bin/env bash
#SBATCH --job-name=gated-baseline-setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/baselines/env.sh"
mkdir -p "$TMPDIR" "$GATED_WORK/archives" "$GATED_WORK/results"
if [[ ! -x "$GATED_ENV/bin/python" ]]; then
    python3 -m venv "$GATED_ENV"
fi
python -m pip freeze > "$GATED_WORK/archives/legacy-environment-$SLURM_JOB_ID.txt"
# Remove the superseded DPR/training stack, which pins incompatible packages.
python -m pip uninstall -y pytorch-lightning torchmetrics faiss-cpu datasets pyarrow pyarrow-hotfix pandas
python -m pip install --upgrade -r "$GATED_ROOT/baselines/requirements.txt"
python -m pip check
python -m pip freeze > "$GATED_WORK/requirements.installed.txt"
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
import os
p = os.environ['GATED_MODEL']
assert AutoConfig.from_pretrained(p, local_files_only=True).model_type == 'qwen3'
t = AutoTokenizer.from_pretrained(p, local_files_only=True)
assert '<think>\n\n</think>' in t.apply_chat_template(
    [{'role':'user','content':'Test'}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
print('Qwen3 tokenizer and non-thinking template verified.')
PY
echo 'Baseline setup completed.'
