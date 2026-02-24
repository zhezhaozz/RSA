# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import os
import sys

sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
from tqdm import tqdm

from .testing_util import run_test

import numpy as np

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def extract_instance_results(results):
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            instance_wise_grades[task_id].append(all([g > 0 for g in generation]))

    instance_wise_grades = [
        v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])
    ]
    return instance_wise_grades

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    in_outs = sample['input_output']
    if isinstance(in_outs, str):
        in_outs = json.loads(in_outs)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(in_outs["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0], metadata_list[0]

if __name__ == "__main__":
    print(
        check_correctness(
            {
                "input_output": json.dumps(
                    {
                        "fn_name": "max_chain_length",
                        "inputs": ["assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 2", "assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4", "assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5"],
                        "outputs": ["3", "4", "5"],
                    },
                )
            },
"""
class Pair(object): 
        def __init__(self, a, b): 
                self.a = a 
                self.b = b 
def max_chain_length(arr, n): 
        max = 0
        mcl = [1 for i in range(n)] 
        for i in range(1, n): 
                for j in range(0, i): 
                        if (arr[i].a > arr[j].b and
                                mcl[i] < mcl[j] + 1): 
                                mcl[i] = mcl[j] + 1
        for i in range(n): 
                if (max < mcl[i]): 
                        max = mcl[i] 
        return max
""",
            6,
            debug=True,
        )
    )
    # print(
    #     check_correctness(
    #         {
    #             "input_output": json.dumps(
    #                 {
    #                     "inputs": [")))))", "(())"],
    #                     "outputs": ["0", "0"],
    #                 },
    #             )
    #         },
    #         "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
    #         6,
    #         debug=True,
    #     )
    # )