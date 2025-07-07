# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Default reward computation function.

    Returns:
        Union[float, Dict[str, Any]]: Either a float score or a dictionary with 'score' and optional 'extra_info'
    """
    if "simplelr_math_35" in data_source or "deepscaler" in data_source:
        from . import hf_math_verify
        res = hf_math_verify.compute_score(solution_str, ground_truth)
    elif "code" in data_source:
        from. import code
        res = code.compute_score(solution_str, ground_truth, extra_info=extra_info)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    if isinstance(res, (int, float, bool)):
        return float(res)
    elif isinstance(res, dict):
        return res
    else:
        return float(res[0])
