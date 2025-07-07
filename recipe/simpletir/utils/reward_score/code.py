import asyncio
import re
import time
import json
import time
import os

import numpy as np

if os.getenv("SANDBOX_ENDPOINT", None) is not None:
    from sandbox.local_sandbox import parallel_sandbox
else:
    from sandbox.internal_sandbox import parallel_sandbox

from recipe.simpletir.agent_utils import truncate_content


MAX_CHAR_DISPLAY = 2048


def compute_score(solution_str, ground_truth, extra_info):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()

    reward_log = []

    pattern = r"```(?:py|python)?\n(.*?)\n```"
    match = re.findall(pattern, solution_str)
    if match:
        solution_code = match[-1].strip()
    else:
        reward_log.append("-" * 16 + "No Code Detected!" + "-" * 16)
        reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
        reward_log.append(solution_str)
        reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
        reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

        reward_log = "\n".join(reward_log)
        reward_log = "❌" * 16 + "Reward Calculation" + "❌" * 16 + "\n" + reward_log + "\n" + "❌" * 16 + f"Final Reward = {0.0}" + "❌" * 16
        print(reward_log + "\n\n")
        return 0.0

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    ground_truth = json.loads(ground_truth)

    t_start = time.time()

    if "functional" in ground_truth:
        code_to_execute = solution_code + "\n" + ground_truth["functional"]
        reward_log.append(code_to_execute)
        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
            parallel_sandbox([code_to_execute], num_processes=256)
        )
        success = sandbox_success[0]
        stdout = str(sandbox_stdout[0])
        stderr = str(sandbox_stderr[0])

        if len(stderr) > 0:
            reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(truncate_content(stdout, max_length=512))
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

            reward_log = "\n".join(reward_log)
            reward_log = "❌" * 16 + "Reward Calculation" + "❌" * 16 + "\n" + reward_log + "\n" + "❌" * 16 + f"Final Reward = {0.0}" + "❌" * 16
            print(reward_log + "\n\n")
            return 0.0
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        reward_log.append(solution_code)
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        tasks = []
        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
            parallel_sandbox(tasks, stdin_list, num_processes=256)
        )

        for stdin, stdout, sandbox_stdout, sandbox_stderr in zip(stdin_list, stdout_list, sandbox_stdout, sandbox_stderr):
            if len(sandbox_stderr) > 0 or sandbox_stdout.strip() != stdout.strip():
                reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                reward_log.append(f"🔎Input: {repr(stdin)}")
                reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                if len(sandbox_stdout) > 0:
                    reward_log.append(
                        f"❌Actual stdout: {truncate_content(sandbox_stdout, max_length=512)}")
                if len(sandbox_stderr) > 0:
                    reward_log.append(
                        f"❌Actual stderr: {truncate_content(sandbox_stderr, max_length=512)}")
                reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

                reward_log = "\n".join(reward_log)
                reward_log = "❌" * 16 + "Reward Calculation" + "❌" * 16 + "\n" + reward_log + "\n" + "❌" * 16 + f"Final Reward = {0.0}" + "❌" * 16
                print(reward_log + "\n\n")
                return 0.0
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    if len(sandbox_stdout) > 0:
        reward_log.append(
            f"stdout: {truncate_content(sandbox_stdout, max_length=512)}")
    if len(sandbox_stderr) > 0:
        reward_log.append(
            f"stderr: {truncate_content(sandbox_stderr, max_length=512)}")
    reward_log = "\n".join(reward_log)
    reward_log = "✅" * 16 + "Reward Calculation" + "✅" * 16 + "\n" + reward_log + "\n" + "✅" * 16 + f"Final Reward = {1.0}" + "✅" * 16
    print(reward_log + "\n\n")
    return 1.0
