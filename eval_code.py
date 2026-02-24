from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional
import pandas as pd
from rewards.code import compute_score
import numpy as np
import random

import datasets
from tqdm import tqdm
import pickle

from pathlib import Path

def load_latest_loop_file(dir_path):
    dir_path = Path(dir_path)

    # Match files of the form loop_{i}.pkl
    pattern = re.compile(r"loop_(\d+)\.pkl$")

    max_i = -1
    latest_file = None

    for file in dir_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                i = int(match.group(1))
                if i > max_i:
                    max_i = i
                    latest_file = file

    if latest_file is None:
        raise FileNotFoundError("No loop_{i}.pkl files found in directory")

    with open(latest_file, "rb") as f:
        data = pickle.load(f)

    return data, max_i, latest_file

def lcb():
    import zlib
    import base64

    def process_fn(example):
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        prompt += f"Question:\n{example['question_content']}"
        if len(example['starter_code']):
            instruction = f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            instruction += f"```python\n{example['starter_code']}\n```\n\n"
        else:
            instruction = f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n\n"
            instruction += f"```python\n# YOUR CODE HERE\n```"

        public_test_cases = json.loads(
            example.pop('public_test_cases')
        )
        private_test_cases = example.pop('private_test_cases')
        try:
            private_test_cases = json.loads(private_test_cases)
        except:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )
        
        eval_types = ["call" if r['testtype'] == "functional" else "stdio" for r in public_test_cases] + ["call" if r['testtype'] == "functional" else "stdio" for r in private_test_cases]
        inputs = [r['input'] for r in public_test_cases] + [r['input'] for r in private_test_cases]
        outputs = [r['output'] for r in public_test_cases] + [r['output'] for r in private_test_cases]
        metadata = json.loads(example['metadata'])

        assert all(x == eval_types[0] for x in eval_types), "Evaluation is a mix of both!"

        return {
            "prompt": prompt,
            'ground_truth': {
                'eval_type': eval_types[0],
                "fn_name": metadata.get("func_name", None),
                'input_output': {
                    "inputs": inputs,
                    "outputs": outputs
                }
            },
            "instruction": instruction
        }

    dataset = datasets.load_dataset("livecodebench/code_generation_lite", version_tag="release_v6", trust_remote_code=True)
    dataset = dataset["test"]
    dataset = dataset.sort("question_id")

    data = list()

    for example in tqdm(dataset):
        data.append(process_fn(example))
    
    return data

def mbpp(split='test'):
    def process_fn(example):
        illustrative_tests = '\n'.join(example['test_list'][:3])
        test_cases = example['test_list'] + example['challenge_test_list']
        test_cases = [example['test_setup_code'] + "\n" + case for case in test_cases]

        instruction = (
            "Reason about the problem and any base cases before writing the code. "
            "You must return the implementation code in the following format:\n"
            "```python\n"
            "<CODE GOES HERE>\n"
            "```\n\n"
        )

        prompt = example["text"] + "\n\n" + "Your code should satisfy these tests:\n\n" + illustrative_tests
        return {
            "prompt": prompt,
            'ground_truth': {
                'input_output': {
                    "inputs": test_cases,
                    "outputs": [None for _ in test_cases]
                },
                'eval_type': 'assert',
            },
            "instruction": instruction
        }

    dataset = datasets.load_dataset("nlile/mbpp")[split]
    data = list()

    for example in tqdm(dataset):
        data.append(process_fn(example))
    return data

def he():
    def process_fn(example):
        prompt = "You will be given a code outline and will generate a correct Python program that matches the specification and passes all tests.\n\n" + f"```python\n{example['prompt']}\n```\n\n"
        example['eval_type'] = 'call'
        instruction = (
            "Reason about the problem and any base cases before writing the code. "
            "You must return the implementation code in the following format:\n"
            "```python\n"
            "<CODE GOES HERE>\n"
            "```\n\n"
        )

        return {
            "prompt": prompt,
            'ground_truth': {
                'eval_type': 'assert',
                'input_output': {
                    "inputs": [example['test'] + "\n\n" + "check(" + example['entry_point'] + ')'],
                    "outputs": [None]
                }
            },
            "instruction": instruction
        }
    
    dataset = datasets.load_dataset("openai/openai_humaneval")['test']
    data = list()

    for example in tqdm(dataset):
        data.append(process_fn(example))
    
    return data

# --------------------- helpers ---------------------
def _append_metrics_to_json(path: str, entry: dict):
    """Append `entry` to a JSON array file at `path` (create if needed)."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                # If somehow not a list, wrap it
                data = [data]
        else:
            data = []
    except Exception:
        # Corrupt or empty file -> start fresh
        data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_question_from_prompt(prompt_cell: Any) -> str:
    """
    Supports a list of chat messages like:
      [{"role": "user", "content": "..."}]
    or a raw string. Returns the first user content when list[dict].
    """
    return prompt_cell[0].get("content", "")

def make_chat_message(question: str) -> str:
    messages = [
        {"role": "user", "content": question},
    ]
    return messages

def make_chat_prompt(tokenizer: AutoTokenizer, messages: list[Dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def render_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    chat_message = make_chat_message(prompt)
    return make_chat_prompt(tokenizer, chat_message)

def aggregate_prompt(question: str, candidate_answers: List[str]) -> str:
    parts = []
    parts.append(
        "You are given a python code implementation problem and several candidate code blocks with their reasoning. "
        "Some candidates may be incorrect or contain errors. "
        "Aggregate the useful ideas and produce a single, high-quality solution. "
        "Reason carefully; if candidates disagree, choose the correct path."
    )
    parts.append(question.strip() + "\n")
    parts.append("Candidate solutions (may contain mistakes):\n")
    for i, ans in enumerate(candidate_answers, 1):
        ans_str = (ans or "").strip()
        parts.append(f"---- Solution {i} ----\n{ans_str}\n")
    parts.append(
        "\nNow provide an improved and correct solution along with its reasoning."
    )
    return "\n".join(parts)

def verify_cot_prompt(question: str, candidate: str) -> str:
    parts = []
    parts.append(
        "You are given a problem and a candidate solution. "
        "Verify whether the candidate solution is correct. "
        "If the solution is correct, output only True. "
        "If it is incorrect, output only False.  "
        "Do not generate anything else. "
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solution:\n")
    parts.append(candidate.strip() + "\n")
    parts.append("Now verify if the solution is True or False. Only output \"True\" or \"False\".")
    return "\n".join(parts)

def build_prompt(tokenizer: AutoTokenizer, question: str, candidate_answers: List[str], instruction: str = None):
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers)
    else:
        prompt = question
    
    prompt += '\n\n' + instruction

    return render_chat_template(tokenizer, prompt)

def verify_candidates(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data: List[dict],
) -> None:
    """
    For each problem, verify each candidate individually and compute mean accuracy among True candidates. If all are False, compute mean acc.
    """
    requests = []
    idxs = []  # (problem_idx, candidate_idx)
    for pi, problem in enumerate(data):
        question = problem['orig_prompt']
        cands = problem.get('candidates') or []
        for ci, cand in enumerate(cands):
            # Build a chat prompt per candidate
            prompt = verify_cot_prompt(question, cand)
            chat_prompt = render_chat_template(tokenizer, prompt)
            requests.append(chat_prompt)
            idxs.append((pi, ci))

    if not requests:
        return

    verify_params = SamplingParams(
        n=1,
        temperature=0.1,#temperature,
        max_tokens=10,
    )
    print(requests[0])
    outs = llm.generate(requests, sampling_params=verify_params)
    all_responses = [o.text for out in outs for o in out.outputs]
    print(all_responses[0])
    verified_vals = [
        1 if (m := re.findall(r'(true|false)', s, flags=re.I)) and m[-1].lower() == "true"
        else 0
        for s in all_responses
    ]
    return verified_vals

def evaluate_k_answers(k_answers: List[str], gt: str) -> Dict[str, Any]:
    """
    Compute per-rollout correctness, mean accuracy, and pass@k against the ground truth.
    Uses the same boxed-extraction logic as your original script.
    """
    correct_bools = [compute_score(e, gt, continuous=False) for e in k_answers]

    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)
    return {
        "pred_accuracies": [int(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
    }

def generate_candidates(A, M, R):
    if A is None:
        return [None for _ in range(M)]

    return [random.sample(A, R) for _ in range(M)]

def reshape_list(lst, K):
    return [lst[i:i+K] for i in range(0, len(lst), K)]

def run(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling: SamplingParams,
    k: int,
    population: int,
    data: List,
    self_verify: bool,
):

    requests, ground_truths = [], []
    for problem in data:
        prompt = problem['orig_prompt']
        ground_truth = problem['ground_truth']
        instruction = problem['instruction']
        candidate_answers = generate_candidates(problem['candidates'], population, k)
        ground_truths.append(ground_truth)
        for candidates in candidate_answers:
            request = build_prompt(tokenizer, prompt, candidates, instruction)
            requests.append(request)
    
    print(requests[0])
    outs = llm.generate(requests, sampling)
    all_responses = [o.text for out in outs for o in out.outputs]
    print(all_responses[0])

    response_length = [len(tokenizer.encode(response)) for response in all_responses]
    median = np.percentile(response_length, 50)
    q25 = np.percentile(response_length, 25)
    q75 = np.percentile(response_length, 75)
    mean_response_length = sum(response_length) / max(1, len(response_length))

    all_responses = reshape_list(all_responses, population)

    for problem, responses in zip(data, all_responses):
        problem['candidates'] = responses

    if self_verify:
        verified_vals = verify_candidates(
                llm, 
                tokenizer, 
                data,
            )
        verified_vals = reshape_list(verified_vals, population)

    # Evaluate
    mean_acc: List[float] = []
    pass_at_k: List[int] = []
    perf_metrics = list()
    verified_score_list: List[float] = []
    correct_bools: List[List[int]] = []

    with tqdm(total=len(all_responses)) as pbar:
        with ProcessPoolExecutor(
            max_workers=48
        ) as executor:
            for gt, responses in zip(ground_truths, all_responses):
                args = (responses, gt)
                perf_metrics.append(executor.submit(evaluate_k_answers, *args))

    assert len(perf_metrics) == len(
        ground_truths
    ), f"results = {len(perf_metrics)} inputs = {len(ground_truths)}"

    perf_metrics = list(perf_metric.result() for perf_metric in perf_metrics)

    for perf_metric in perf_metrics:
        mean_acc.append(perf_metric['mean_acc'])
        pass_at_k.append(perf_metric['pass_at_k'])
        correct_bools.append(perf_metric['pred_accuracies'])

    if self_verify:
        for bools, verified in zip(correct_bools, verified_vals):
            verified_score = sum([x*y for x,y in zip(bools, verified)]) / max(1, sum(verified))
            verified_score_list.append(verified_score)

    metrics = json.dumps(
        {
            "n_samples": len(mean_acc),
            "k": k,
            "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
            "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
            "self_verified_acc": sum(verified_score_list) / max(1, len(verified_score_list)),
            "mean_length": mean_response_length,
            "median_length": median,
            "q25_length": q25,
            "q75_length": q75,
        }, indent=2
    )
    return data, metrics

def loop(
    model_name: str,
    loops: int,
    k: int,
    population: int,
    seed_dataset: str,
    output_dir: str,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
    self_verify: bool,
    resume: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if 'nemo' in model_name:
        llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                    dtype=dtype, trust_remote_code=True, seed=seed,
                    mamba_ssm_cache_dtype='float32')
    else:
        llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                    dtype=dtype, trust_remote_code=True, seed=seed)

    sampling = SamplingParams(
        n=1, temperature=temperature, max_tokens=max_new_tokens
    )

    # write aggregated per-loop metrics (lists + mean/std), path unchanged
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir,'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed)+'.json')
    if not resume:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
    checkpoints_path = os.path.join(output_dir, 'checkpoints/' + 'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed))
    os.makedirs(checkpoints_path, exist_ok=True)

    if seed_dataset == 'lcb':
        data = lcb()
    elif seed_dataset == 'mbpp':
        data = mbpp()
    elif seed_dataset == 'he':
        data = he()

    # control RNG for candidate sampling too
    random.seed(seed)
    np.random.seed(seed)

    if resume:
        try:
            data, start_loop_idx, _ = load_latest_loop_file(checkpoints_path)
        except:
            print(f'Checkpoint not found; defaulting to base')
            data = [
                {
                    'orig_prompt':  row['prompt'],
                    'ground_truth': row['ground_truth'],
                    'candidates': None,
                    'instruction': row['instruction']
                }
                for row in data
            ]
            start_loop_idx = -1
    else:
        data = [
            {
                'orig_prompt':  row['prompt'],
                'ground_truth': row['ground_truth'],
                'candidates': None,
                'instruction': row['instruction']
            }
            for row in data
        ]
        start_loop_idx = -1
            
    for loop_idx in range(start_loop_idx + 1, loops):
        data, metrics = run(
            llm=llm,
            tokenizer=tokenizer,
            sampling=sampling,
            k=k,
            population=population,
            data=data,
            self_verify=self_verify,
        )
        with open(os.path.join(checkpoints_path,f'loop_{loop_idx}.pkl'), 'wb') as file:
            pickle.dump(data, file)

        print(loop_idx, metrics)
        metrics_dict = json.loads(metrics)
        out_entry = {
            "n_samples": metrics_dict.get("n_samples", None),
            "k": k,
            "population": population,
            "loop": loop_idx,
            "mean_acc_k": metrics_dict["mean_acc_k"],
            "mean_pass_at_k": metrics_dict["mean_pass_at_k"],
            "self_verified_acc": metrics_dict["self_verified_acc"],
            "mean_length": metrics_dict["mean_length"],
            "median_length": metrics_dict["median_length"],
            "q25_length": metrics_dict["q25_length"],
            "q75_length": metrics_dict["q75_length"],
        }

        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="lcb")
    ap.add_argument("--output", default="eval/")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action='store_true', default=False)
    ap.add_argument("--self_verify", action='store_true', default=False)
    args = ap.parse_args()

    loop(
        model_name=args.model,
        loops=args.loops,
        seed_dataset=args.dataset,
        output_dir=os.path.join(args.output, args.model.split('/')[-1]),
        k=args.k,
        population=args.population,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
        resume=args.resume,
        self_verify=args.self_verify,
    )

if __name__ == "__main__":
    main()
