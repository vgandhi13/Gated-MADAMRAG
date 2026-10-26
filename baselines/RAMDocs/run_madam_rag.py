import argparse
import os
import re
import json
import torch
import string
from tqdm import tqdm
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def call_llm(prompt: str, generator, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": prompt}]
    output = generator(
                messages,
                max_new_tokens=max_new_tokens,
                top_p=None,
                do_sample=False)
    return output[0]["generated_text"][-1]['content'].strip()


def agent_response(query: str, document: str, generator, history: str = ""):
    if history:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following reponses are from other agents as additional information.
{history}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.  
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""
    else:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

    output = call_llm(prompt, generator)
    return output


def aggregate_responses(query: str, responses: List[str], generator):
    joined = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
    prompt = f"""You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963. 
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{joined}
"""
    return call_llm(prompt, generator)


def multi_agent_debate(query: str, documents: List[str], generator, num_rounds: int = 3):
    records = {}
    num_agents = len(documents)
    agent_outputs = []

    # Round 1
    records["round1"] = {"answers": [], "explanations": []}
    for doc in documents:
        response = agent_response(query, doc, generator)
        answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
        explanation = response[response.find("Explanation: ") + len("Explanation: "):]
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)
    records["round1"]["aggregation"] = aggregate_responses(query, agent_outputs, generator)

    # Additional rounds
    final_aggregation = None
    for t in range(1, num_rounds):
        round_key = f"round{t+1}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs = []
        for i, doc in enumerate(documents):
            history = "\n".join([f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i])
            response = agent_response(query, doc, generator, history)
            answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
            explanation = response[response.find("Explanation: ") + len("Explanation: "):]
            records[round_key]["answers"].append(answer)
            records[round_key]["explanations"].append(explanation)
            new_outputs.append(response)
        agent_outputs = new_outputs
        pred_ans_list = []
        for ans in records[round_key]["answers"]:
            pred_ans_list.append(normalize_answer(ans))
        prev_pred_ans_list = []
        for ans in records[f"round{t}"]["answers"]:
            prev_pred_ans_list.append(normalize_answer(ans))
        assert len(pred_ans_list) == len(prev_pred_ans_list)
        flag = True
        for k in range(len(pred_ans_list)):
            if pred_ans_list[k] in prev_pred_ans_list[k] or prev_pred_ans_list[k] in pred_ans_list[k]:
                continue
            else:
                flag = False
        if flag:
            final_aggregation = records[f"round{t}"]["aggregation"]
            break
        else:
            records[round_key]["aggregation"] = aggregate_responses(query, agent_outputs, generator)
            final_aggregation = records[round_key]["aggregation"]

    records["final_aggregation"] = final_aggregation
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hf_token = os.getenv('HF_TOKEN', None)
    args.output_path = f"{args.data_path}_madam_rag_{args.model_name.split('/')[-1]}_rounds{args.num_rounds}.jsonl"

    set_seed(42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, token=hf_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True, device_map="auto")

    with open(args.data_path, "r") as f:
        all_data = [json.loads(line.strip()) for line in f]

    results = []
    for i in tqdm(range(len(all_data)), desc="Running MADAM-RAG"):
        entry = all_data[i]
        documents = [doc["text"] for doc in entry["documents"]]
        result = multi_agent_debate(entry["question"], documents, generator, num_rounds=args.num_rounds)
        results.append(result)

    with open(args.output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()