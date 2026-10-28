"""Single generation over all supplied passages; no retrieval or indexing."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.data import render_document

ANSWER_INSTRUCTION = (
    'Answer the question. If the question is ambiguous, provide all distinct correct '
    'answers. Exclude incorrect or misleading answers. If no answer is known, use an '
    'empty list. Return only this format, with short answer strings: '
    'All Correct Answers: ["answer 1", "answer 2"].'
)


def build_prompt(question, documents):
    passages = '\n\n'.join(f'Document {i + 1}: {render_document(doc)}'
                            for i, doc in enumerate(documents))
    return (f'{ANSWER_INSTRUCTION}\nUse the following retrieved evidence to answer.\n\n'
            f'Question: {question}\n\n{passages}\n\nAll Correct Answers:')


if __name__ == '__main__':
    from baselines.run import main
    main(default_baseline='standard_rag')
