"""Single generation from the question and parametric knowledge only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.StandardRAG.run_standard_rag import ANSWER_INSTRUCTION


def build_prompt(question):
    return f'{ANSWER_INSTRUCTION}\n\nQuestion: {question}\n\nAll Correct Answers:'


if __name__ == '__main__':
    from baselines.run import main
    main(default_baseline='no_rag')
