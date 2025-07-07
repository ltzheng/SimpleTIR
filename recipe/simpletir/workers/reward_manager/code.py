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

import asyncio
from typing import Dict, List

import torch

from verl import DataProto
from recipe.simpletir.utils.reward_score import _default_compute_score
from multiprocessing import cpu_count


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, timeout=300., semaphore=None):
    if semaphore is None:
        raise ValueError("Semaphore must be provided for async execution.")
    
    async with semaphore:
        try:
            # Ensure process_completion is called properly
            result = await asyncio.wait_for(
                evaluation_func(task, completion, reference, task_extra_info),
                timeout=timeout
            )
            # merge result with extra_info if available
            if 'extra_info' in result:
                result['extra_info'].update({"is_filter": 0})  # by default we will not filter out these cases
            elif not isinstance(result, dict):
                result = {"score": result, "extra_info": {"is_filter": 0}}  # by default we will not filter out these cases
            return result
        except asyncio.TimeoutError:
            # For timeout errors, we will filter out these cases in loss calculation
            print(f"Timeout occurred for completion: {completion[:64]}...")
            return {"score": 0, "extra_info": {"is_filter": 1}}  # by default we will filter out timeout cases
        except Exception as e:
            # TODO: we do not know if we should filter out these cases 
            print(f"Error processing completion: {completion[:10]}, Error: {e}")
            return {"score": 0, "extra_info": {"is_filter": 0}}


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_infos=None,
                                       num_processes=64):
    semaphore = asyncio.Semaphore(num_processes)
    # resize extra_infos
    if extra_infos is None:
        extra_infos = [None] * len(tasks)
    tasks = [
        single_compute_score(
            evaluation_func,
            completion, 
            reference, 
            task, 
            task_extra_info,
            timeout=40.,
            semaphore=semaphore
        )
        for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_infos)
    ]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        # if all failed, we have to skip this batch
        print(f"Unexpected error in parallel requests: {e}")
        results = [{"score": 0, "extra_info": {"is_filter": 1}}] * len(completions)
    return results


class CodeRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        extra_info_dict: Dict[str, List[float]] = {}

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        
        response_ids = data.batch['responses']
        valid_response_lengths = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)

        # (NOTE) qian: since we have already passed the special token and we remove everything after EOS,
        #  here we can safely decode them directly as the string
        prompt_strs = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        response_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]

        num_processes = max(cpu_count() - 16, 1) # we keep 16 cpus in case they are being used by other processes
        assert len(response_strs) == len(ground_truths) == len(data_sources)
        try:
            results = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             response_strs,
                                             ground_truths,
                                             data_sources,
                                             extra_infos=extra_infos,
                                             num_processes=num_processes))
        except asyncio.TimeoutError as e:
            # becasue of timeout, we set all as 0.
            print(f"Unexpected timeout in batched reward computing. Setting all as 0: {e}")
            results = [{"score": 0, "extra_info": {"is_filter": 1}}] * len(response_strs)
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0: {e}")
            results = [{"score": 0, "extra_info": {"is_filter": 1}}] * len(response_strs)
        
        scores = [result['score'] for result in results]
        
        for i, item_result in enumerate(results):
            for key, value in item_result['extra_info'].items():
                if key not in extra_info_dict:
                    extra_info_dict[key] = [0.0] * len(scores)
                extra_info_dict[key][i] = value

        # we should guarantee that the results are in the same format
        assert len(scores) == len(response_strs)
        
        # Print examples and set rewards
        for i in range(len(data)):
            data_item = data[i]
            data_source = data_sources[i]
            score = scores[i]
            
            reward_tensor[i, valid_response_lengths[i].item() - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return {'reward_tensor': reward_tensor, 'extra_info': extra_info_dict}
