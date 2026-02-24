# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import traceback

from .code_utils import check_correctness
import re

pattern = r"```python(.*?)```"

def extract_solution(text: str) -> str | None:
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return text

def compute_score(solution_str, ground_truth, continuous: bool = False) -> float:
    code = extract_solution(solution_str).strip()
    if isinstance(ground_truth, str):
        ground_truth = json.load(ground_truth)
    
    try:
        result, metadata = check_correctness(ground_truth, code, 6, False)
    except Exception as e:
        result = [-2]
        metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }

    if continuous:
        return sum([1 if obj is True else 0 for obj in result]) / len(result)
    else:
        return float(all([True if obj is True else False for obj in result]))
