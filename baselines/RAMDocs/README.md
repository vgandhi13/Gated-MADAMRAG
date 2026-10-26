# [Retrieval-Augmented Generation with Conflicting Evidence](https://arxiv.org/abs/2504.13079)
by [Han Wang](https://hannight.github.io/), [Archiki Prasad](https://archiki.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/).
* 🤗 [**RAMDocs Dataset**](https://huggingface.co/datasets/HanNight/RAMDocs)

![Image](https://github.com/user-attachments/assets/d80f8455-0c00-4c67-b570-885c3aa3f992)


## RAMDocs
RAMDocs is a dataset that simulates complex and realistic scenarios for conflicting evidence for a user query, including ambiguity, misinformation, and noise. We provide the raw data file `RAMDocs_test.jsonl`.

Each instance contains the following fields:
- `question`: The question
- `documents`: list of documents, where each document contains the following fields:
    - `text`: text of the document
    - `type`: the type of the document, which can be one of the following:
        - `correct`: the document contains the correct answer to the question
        - `misinfo`: the document contains misinformation, which is a wrong answer to the question
        - `noise`: the document does not provide the answer to the question
    - `answer`: the answer to the question, which can be infered from the document. When the type is `noise`, the answer is `unknown`
- `disambig_entity`: list of disambiguous entities that share the same ambiguous name in the question
- `gold_answers`: list of gold answers for different disambiguous entities
- `wrong_answers`: list of wrong answers to the question

The following figure shows the summary statistics across key dimensions, including the number of correct and incorrect answers per example, the total number of documents retrieved, and the distribution of documents that support correct answers, incorrect answers, or contain irrelevant noise.
![Image](https://github.com/user-attachments/assets/d02873cb-c845-4d47-a9c2-1829e1f34bc6)

## MADAM-RAG
MADAM-RAG is a structured, multi-agent framework designed to handle inter-document conflicts, misinformation, and noise in retrieved content.
Our overall framework consists of three key components:
- independent LLM dialogue agents that generate intermediate responses conditioned on a single document.
- a centralized aggregator.
- an iterative multi-round debate process.
![Image](https://github.com/user-attachments/assets/0c206cab-6742-4b55-b902-c6e99b191683)

### Requirements
You can install all required packages by running the following command:
```bash
pip install -r requirements.txt
```

### Run MADAM-RAG
To run the MADAM-RAG framework, you can use the following example command:
```bash
CUDA_VISIBLE_DEVICES=0
HF_TOKEN=your_huggingface_token
python run_madam_rag.py \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path RAMDocs_test.jsonl \
```

We explain arguments below (some are not shown in the example):
- `model_name`: The name of the model to use
- `cache_dir`: The directory to cache the model
- `data_path`: The path to the data file
- `num_rounds`: The maximum number of rounds to run the debate process
- `seed`: The random seed to use


## Aknowledgement
We sincerely thank the authors of [AmbigDocs](https://arxiv.org/abs/2404.12447) for their public data release.

## Citation
```bibtex
@article{wang2025retrieval,
  title={Retrieval-Augmented Generation with Conflicting Evidence},
  author={Han Wang and Archiki Prasad and Elias Stengel-Eskin and Mohit Bansal},
  year={2025}
  journal={arXiv preprint arXiv:2504.13079}
}
```