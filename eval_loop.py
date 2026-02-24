from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional, Callable
import pandas as pd
from rewards.math import last_boxed_only_string, remove_boxed, is_equiv
import numpy as np
import random
from reasoning_gym.factory import get_score_answer_fn
from datasets import Dataset
from tqdm import tqdm
import pickle
from pathlib import Path
from functools import partial

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

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


def extract_rg_solution(completion: str) -> Optional[str]:
    """Extract the model's predicted answer for reasoning-gym style prompts.

    Priority order:
    1. Anything enclosed by <answer> ... </answer> (preferred new format).
    2. Text following "Final Answer:" after an optional "</think>" tag (legacy format).
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    parts = completion.split("</think>", 1)
    if len(parts) == 1:
        return None

    tail = parts[1].strip()
    final_response = tail.rsplit("Final Answer:", 1)
    if len(final_response) == 1:
        return None

    return final_response[1].strip()


# make sure to include all the possible data sources in the if-else
def get_task_name(ds: Dataset) -> str:
    data_source = ds[0]['data_source']
    if "aime" in data_source or "hmmt" in data_source or "MATH" in data_source or "DeepScaleR" in data_source:
        return "math"
    elif "reasoning_gym" in data_source:
        return "rg"
    elif "m-a-p/SuperGPQA" in data_source:
        return 'supergpqa'
    elif data_source == 'lcb':
        return 'code'
    else:
        raise ValueError(f"Unknown task: {data_source}")


# --------------------- prompt building ---------------------
def render_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    chat_message = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)


def render_chat_template_gpt(tokenizer: AutoTokenizer, prompt: str, reasoning) -> str:
    convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(reasoning)),
        Message.from_role_and_content(Role.USER, prompt),
    ]
    )

    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    return prefill_ids


def aggregate_prompt(question: str, candidate_answers: List[str], task: str) -> str:
    if task == 'rg':
        problem_kind = 'problem'
        format_hint = '<answer>...</answer>'
    elif task == 'supergpqa':
        problem_kind = 'multiple-choice problem'
        format_hint = '\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}'
    else:
        problem_kind = 'math problem'
        format_hint = '\\boxed{}'

    parts = []
    if len(candidate_answers) == 1:
        parts.append(
            f"You are given a {problem_kind} and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            f"End with the final result in {format_hint}.\n"
        )
    else:
        parts.append(
            f"You are given a {problem_kind} and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy."
            f"End with the final result in {format_hint}.\n"
        )

    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")

    if len(candidate_answers) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        ans_str = (candidate_answers[0] or "").strip()
        parts.append(f"---- Candidate ----\n{ans_str}\n")
        parts.append(
            f"Now refine the candidate into an improved solution. Provide clear reasoning and end with the final answer in {format_hint}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, ans in enumerate(candidate_answers, 1):
            ans_str = (ans or "").strip()
            parts.append(f"---- Solution {i} ----\n{ans_str}\n")
        parts.append(
            f"Now write a single improved solution. Provide clear reasoning and end with the final answer in {format_hint}."
        )

    return "\n".join(parts)


def build_prompt(tokenizer: AutoTokenizer, question: str, candidate_answers: Optional[List[str]], task: str, chat_template_fn: Callable[[AutoTokenizer, str], List]):
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers, task)
    else:
        prompt = question
    return chat_template_fn(tokenizer, prompt)


# --------------------- summarization ---------------------
def summarize_cot_prompt(question: str, candidate: str) -> str:
    parts = []
    parts.append(
        "You are given a math problem and a candidate solution. "
        "Summarize the solution into a concise chain-of-thought style outline that preserves all "
        "important information required to continue refinement later: the main approach(es), key steps/equations, "
        "useful intermediate results, and any mistakes or dead ends. "
        "Compress it while keeping the essential structure. "
        "If the candidate included a final answer, retain it at the end in \\boxed{ }.\n"
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solution:\n")
    parts.append(candidate.strip() + "\n")
    parts.append("Now produce the concise, information-preserving summary. "
                 "End with the final answer in \\boxed{} if present.")
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

def summarize_candidates_inplace(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data: List[dict],
    max_tokens: int,
    temperature: float,
    chat_template_fn: Callable[[AutoTokenizer, str], List],
    prompt_token_ids: bool = False,
) -> None:
    """
    For each problem, summarize each candidate individually and replace in place.
    """
    requests = []
    idxs = []  # (problem_idx, candidate_idx)
    for pi, problem in enumerate(data):
        question = problem['orig_prompt']
        cands = problem.get('candidates') or []
        for ci, cand in enumerate(cands):
            # Build a chat prompt per candidate
            prompt = summarize_cot_prompt(question, cand)
            chat_prompt = chat_template_fn(tokenizer, prompt)
            requests.append(chat_prompt)
            idxs.append((pi, ci))

    if not requests:
        return

    if prompt_token_ids:
        summarize_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids
        )
    else:
        summarize_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens
        )
    if prompt_token_ids:
        outs = llm.generate(prompt_token_ids=requests, sampling_params=summarize_params)
    else:
        outs = llm.generate(requests, sampling_params=summarize_params)
    flat = [o.text for out in outs for o in out.outputs]

    # Write summaries back in place
    for (pi, ci), summary in zip(idxs, flat):
        data[pi]['candidates'][ci] = summary


# --------------------- evaluation ---------------------
def verify_candidates(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data: List[dict],
    chat_template_fn: Callable[[AutoTokenizer, str], List],
    prompt_token_ids: bool = False,    
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
            chat_prompt = chat_template_fn(tokenizer, prompt)
            requests.append(chat_prompt)
            idxs.append((pi, ci))

    if not requests:
        return

    if prompt_token_ids:
        summarize_params = SamplingParams(
            n=1,
            temperature=0.1,#temperature,
            max_tokens=10,
            stop_token_ids=stop_token_ids
        )
        print(tokenizer.decode(requests[0]))
        outs = llm.generate(prompt_token_ids=requests, sampling_params=summarize_params)
    else:
        summarize_params = SamplingParams(
            n=1,
            temperature=0.1,#temperature,
            max_tokens=10,
        )
        print(requests[0])
        outs = llm.generate(requests, sampling_params=summarize_params)

    all_responses = [o.text for out in outs for o in out.outputs]

    print(all_responses[0])
    verified_vals = [
        1 if (m := re.findall(r'(true|false)', s, flags=re.I)) and m[-1].lower() == "true"
        else 0
        for s in all_responses
    ]
    return verified_vals


def evaluate_k_answers_math(k_answers: List[str], gt: str) -> Dict[str, Any]:
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in k_answers
    ]
    extracted = [remove_boxed(s) for s in solutions]

    ## mean accuracy, pass@k
    correct_bools = [bool(is_equiv(e, gt)) for e in extracted]
    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)

    ## majority vote
    clusters: List[Dict[str, Any]] = []
    for e in extracted:
        placed = False
        for c in clusters:
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        majority_vote = 0.0

    best = max(clusters, key=lambda c: c["count"])
    majority_vote = float(bool(is_equiv(best["rep"], gt)))

    return {
        "pred_accuracies": [float(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


def evaluate_k_answers_rg(score_answer_fn: Callable[[str, str], float], k_answers: List[str], gt: dict) -> Dict[str, Any]:
    solutions = [extract_rg_solution(a) or "" for a in k_answers]

    ## mean accuracy, pass@k

    scores = []
    for sol in solutions:
        try:
            scores.append(score_answer_fn(sol, gt))
        except:
            scores.append(0)

    # scores = [float(score_answer_fn(sol, gt)) for sol in solutions]
    mean_acc = float(sum(scores) / max(1, len(scores)))
    pass_at_k = float(1.0 if any(s == 1.0 for s in scores) else 0.0)

    ## majority vote
    clusters: List[Dict[str, Any]] = []
    for sol in solutions:
        placed = False
        for c in clusters:
            if bool(is_equiv(sol, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": sol, "count": 1})

    if not clusters:
        majority_vote = 0.0

    best = max(clusters, key=lambda c: c["count"])
    majority_vote = float(score_answer_fn(best["rep"], gt))

    return {
        "pred_accuracies": [float(s) for s in scores],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


# --------------------- main ---------------------
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
    task: str,
    self_verify: bool,
    chat_template_fn: Callable[[AutoTokenizer, str], List],
    prompt_token_ids: bool = False,
    score_answer_fn: Optional[Callable[[str, str], float]] = None,
):

    requests, ground_truths, dataset_names = [], [], []
    for problem in data:
        prompt = problem['orig_prompt']
        ground_truth = problem['gt']
        candidate_answers = generate_candidates(problem['candidates'], population, k)
        ground_truths.append(ground_truth)
        dataset_names.append(problem['dataset_name'])
        for candidates in candidate_answers:
            request = build_prompt(tokenizer, prompt, candidates, task, chat_template_fn=chat_template_fn)
            requests.append(request)
    
    if prompt_token_ids:
        print(tokenizer.decode(requests[0]))
        outs = llm.generate(prompt_token_ids=requests, sampling_params=sampling)
    else:
        print(requests[0])
        outs = llm.generate(requests, sampling_params=sampling)
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
                chat_template_fn=chat_template_fn,
                prompt_token_ids=prompt_token_ids
            )
        verified_vals = reshape_list(verified_vals, population)

    # Evaluate
    mean_acc: List[float] = []
    pass_at_k: List[float] = []
    majority_acc: List[float] = []
    verified_score_list: List[float] = []
    correct_bools = []

    for dataset_name, gt, responses in zip(dataset_names, ground_truths, all_responses):
        if task == 'rg':
            score_answer_fn = get_score_answer_fn(name=dataset_name)
            perf_metric = evaluate_k_answers_rg(score_answer_fn, responses[:], gt)
        else:
            perf_metric = evaluate_k_answers_math(responses[:], gt) # Also works for supergpqa
        mean_acc.append(perf_metric['mean_acc'])
        pass_at_k.append(perf_metric['pass_at_k'])
        majority_acc.append(perf_metric['majority_vote_correct'])
        correct_bools.append(perf_metric['pred_accuracies'])

    if self_verify:
        for dataset_name, gt, responses, verified in zip(dataset_names, ground_truths, all_responses, verified_vals):
            if task == 'rg':
                score_answer_fn = get_score_answer_fn(name=dataset_name)
                solutions = [extract_rg_solution(a) or "" for a in responses[:]]
                scores = [float(score_answer_fn(sol, gt)) for sol in solutions]
            else:
                solutions = [
                    (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
                    for a in responses
                ]
                extracted = [remove_boxed(s) for s in solutions]
                scores = [float(is_equiv(e, gt)) for e in extracted]
            verified_score = sum([x*y for x,y in zip(scores, verified)]) / max(1, sum(verified))
            verified_score_list.append(verified_score)

    
    metrics = json.dumps(
        {
            "n_samples": len(mean_acc),
            "k": k,
            "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
            "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
            "mean_majority_acc": sum(majority_acc) / max(1, len(majority_acc)),
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
    summarize_cot: bool,
    seed_dataset: str,
    output_dir: str,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
    resume: bool,
    remove_checkpoint: bool,
    reasoning: str,
    self_verify: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if 'nemo' in model_name:
        llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                    dtype=dtype, trust_remote_code=True, seed=seed,
                    mamba_ssm_cache_dtype='float32')
    else:
        llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                    dtype=dtype, trust_remote_code=True, seed=seed)
    if 'gpt' in model_name:
        sampling = SamplingParams(
            n=1, temperature=temperature, max_tokens=max_new_tokens, stop_token_ids=stop_token_ids
        )
    else:
        sampling = SamplingParams(
            n=1, temperature=temperature, max_tokens=max_new_tokens
        )
    ds = Dataset.from_parquet(seed_dataset)

    if 'gpt' in model_name:
        if reasoning == 'low':
            reasoning = ReasoningEffort.LOW
        elif reasoning == 'medium':
            reasoning = ReasoningEffort.MEDIUM
        elif reasoning == 'high':
            reasoning = ReasoningEffort.HIGH
        else:
            reasoning = None

        chat_template_fn = partial(render_chat_template_gpt, reasoning=reasoning)
    else:
        chat_template_fn = render_chat_template

    # Prepare scorer for RG when needed
    score_answer_fn: Optional[Callable[[str, str], float]] = None
    task = get_task_name(ds)

    # control RNG for candidate sampling too
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed)+'.json')
    if not resume:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
    checkpoints_path = os.path.join(output_dir, 'checkpoints/' + 'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed))
    os.makedirs(checkpoints_path, exist_ok=True)

    if resume:
        try:
            data, start_loop_idx, _ = load_latest_loop_file(checkpoints_path)
            print(f'Starting Inference from Loop: {start_loop_idx + 1}')
        except:
            print(f'Checkpoint not found; defaulting to base')
            data = [
                {
                    'orig_prompt': extract_question_from_prompt(row['prompt']),
                    'dataset_name': (row['extra_info']['dataset_name'] if task == 'rg' else None),
                    'gt': (json.loads(row['extra_info']['entry']) if task == 'rg' else row['reward_model']['ground_truth']),
                    'candidates': None,
                }
                for row in ds
            ]
            start_loop_idx = -1
    else:
        data = [
            {
                'orig_prompt': extract_question_from_prompt(row['prompt']),
                'dataset_name': (row['extra_info']['dataset_name'] if task == 'rg' else None),
                'gt': (json.loads(row['extra_info']['entry']) if task == 'rg' else row['reward_model']['ground_truth']),
                'candidates': None,
            }
            for row in ds
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
            task=task,
            score_answer_fn=score_answer_fn,
            self_verify=self_verify,
            chat_template_fn=chat_template_fn,
            prompt_token_ids=True if 'gpt' in model_name else False
        )
        with open(os.path.join(checkpoints_path,f'loop_{loop_idx}.pkl'), 'wb') as file:
            pickle.dump(data, file)

        print(loop_idx, metrics)
        if summarize_cot and loop_idx < loops - 1:
            print("Summarizing candidates before aggregation...")
            summarize_candidates_inplace(
                llm=llm,
                tokenizer=tokenizer,
                data=base_structure,
                max_tokens=max_new_tokens,
                temperature=temperature,
                chat_template_fn=chat_template_fn,
                prompt_token_ids=True if 'gpt' in model_name else False
            )
        metrics_dict = json.loads(metrics)

        out_entry = {
            "n_samples": metrics_dict.get("n_samples", None),
            "k": k,
            "population": population,
            "loop": loop_idx,
            "task": task,
            "mean_acc_k": metrics_dict["mean_acc_k"],
            "mean_pass_at_k": metrics_dict["mean_pass_at_k"],
            "mean_majority_acc": metrics_dict["mean_majority_acc"],
            "self_verified_acc": metrics_dict["self_verified_acc"],
            "mean_length": metrics_dict["mean_length"],
            "median_length": metrics_dict["median_length"],
            "q25_length": metrics_dict["q25_length"],
            "q75_length": metrics_dict["q75_length"],
        }

        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")
    
    if remove_checkpoint:
        shutil.rmtree(checkpoints_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="./data/aime25/train.parquet")
    ap.add_argument("--output", default="./eval")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--summarize-cot", action="store_true")
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action='store_true', default=False)
    ap.add_argument("--remove_checkpoint", action='store_true', default=False)
    ap.add_argument("--self_verify", action='store_true', default=False)
    ap.add_argument("--reasoning", default="medium")
    args = ap.parse_args()

    loop(
        model_name=args.model,
        loops=args.loops,
        seed_dataset=args.dataset,
        output_dir=os.path.join(args.output, args.model.split('/')[-1]),
        k=args.k,
        population=args.population,
        summarize_cot=args.summarize_cot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
        resume=args.resume,
        remove_checkpoint=args.remove_checkpoint,
        self_verify=args.self_verify,
        reasoning=args.reasoning
    )


if __name__ == "__main__":
    main()