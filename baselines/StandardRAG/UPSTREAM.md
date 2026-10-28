The previous Hugging Face DPR/BART RAG example was from
https://github.com/huggingface/transformers-research-projects.git
commit fccd0cdab03c1407bb7def05e1b48528fc817078, subdirectory rag/.
Its Apache-2.0 license is retained in LICENSE.

That pipeline performed fresh index retrieval, which is not the concatenated-prompt
baseline in this project's proposal. Its retrieval/training code was removed.
An archive made before cleanup is kept outside this repository
(gated-standard-rag-legacy-20261028.tar.gz).

run_standard_rag.py is the local implementation over saved retrieved evidence.
