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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pprint import pprint
from typing import Dict, Type

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from recipe.simpletir.agent_utils import AgentHelper, GenerationConfig
from recipe.simpletir.turn_predictor import TurnPredictorConfig
from recipe.simpletir.utils.dataset.rl_dataset import RLCustomPromptDataset
from verl import DataProto
from verl.protocol import DataProtoItem, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format
        meta_info=item.meta_info,
    )


import torch

from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(
    data: DataProto,
    kl_ctrl: core_algos.AdaptiveKLController,
    kl_penalty="kl",
    mask_tool_output=True,
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    if mask_tool_output:
        attention_mask = data.batch["info_mask"]
    else:
        attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }

    return data, metrics


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                .detach()
                .item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(
            torch.eq(prompt_length, max_prompt_length).float()
        )
        .detach()
        .item(),
    }

    # metrics for actions
    if "turns_stats" in batch.meta_info:
        metrics["env/num_turns/mean"] = float(
            np.array(batch.meta_info["turns_stats"], dtype=np.int16).mean()
        )
        metrics["env/num_turns/max"] = float(
            np.array(batch.meta_info["turns_stats"], dtype=np.int16).max()
        )
        metrics["env/num_turns/min"] = float(
            np.array(batch.meta_info["turns_stats"], dtype=np.int16).min()
        )
        # ratio of samples with 1 turn
        metrics["env/num_one_turn"] = np.array(
            batch.meta_info["turns_stats"], dtype=np.int16
        ).tolist().count(1) / len(batch.meta_info["turns_stats"])
    if "active_mask" in batch.meta_info:
        metrics["env/finish_ratio"] = 1 - float(
            np.array(batch.meta_info["active_mask"], dtype=np.int16).mean()
        )
    if "void_turn_mask" in batch.meta_info:
        metrics["env/void_turn_ratio"] = 1 - float(
            np.array(batch.meta_info["void_turn_mask"], dtype=np.int16).mean()
        )
    if "use_code_stats" in batch.meta_info:
        metrics["env/num_code_use"] = float(
            np.array(batch.meta_info["use_code_stats"], dtype=np.int16).mean()
        )
        metrics["env/code_use_ratio"] = float(
            (
                np.array(batch.meta_info["use_code_stats"], dtype=np.int16)
                / np.array(batch.meta_info["turns_stats"], dtype=np.int16)
            ).mean()
        )
    if "valid_code_stats" in batch.meta_info:
        metrics["env/num_valid_code"] = float(
            np.array(batch.meta_info["valid_code_stats"], dtype=np.int16).mean()
        )
        metrics["env/valid_code_ratio"] = float(
            (
                np.array(batch.meta_info["valid_code_stats"], dtype=np.int16)
                / np.array(batch.meta_info["turns_stats"], dtype=np.int16)
            ).mean()
        )
    if "success_code_lines" in batch.meta_info:
        metrics["env/success_code_lines"] = float(
            np.array(batch.meta_info["success_code_lines"], dtype=np.int16).mean()
        )
        metrics["env/fail_code_lines"] = float(
            np.array(batch.meta_info["fail_code_lines"], dtype=np.int16).mean()
        )
        metrics["env/success_code_strip_lines"] = float(
            np.array(batch.meta_info["success_code_strip_lines"], dtype=np.int16).mean()
        )
        metrics["env/fail_code_strip_lines"] = float(
            np.array(batch.meta_info["fail_code_strip_lines"], dtype=np.int16).mean()
        )

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name]
            * 1000
            / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RaySimpleTIRTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # register wandb generation logger
        self.validation_generations_logger = {}
        val_file_prefix = os.environ.get("DATA_PATH", "")
        for val_file in config.data.val_files:
            val_file = val_file.removeprefix(val_file_prefix + "/").split(".")[0]
            self.validation_generations_logger[val_file] = ValidationGenerationsLogger(
                val_file
            )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                config.algorithm.kl_ctrl
            )

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert (
                config.data.train_batch_size
                >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )

        if (
            config.algorithm.use_kl_in_reward
            and config.actor_rollout_ref.actor.use_kl_loss
        ):
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
                > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1)
                > 1
            ):
                assert config.actor_rollout_ref.model.use_remove_padding, (
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.val_kwargs.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLCustomPromptDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            prompt=self.config.data.prompt,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            apply_chat_template=self.config.data.apply_chat_template,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
        )
        assert self.train_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), (
            f"dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get('truncation', 'error')}"
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.data.train_batch_size
        # rejection sampling or remove clipped length both require a larger batch size in prior
        if self.config.trainer.rejection_sample or self.config.trainer.remove_clip:
            train_batch_size *= self.config.trainer.oversample_multiplier
            # round train_batch_size to the multipler of world size
            world_size = (
                self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
            )
            train_batch_size = int(train_batch_size // world_size * world_size)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLCustomPromptDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            prompt=self.config.data.prompt,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            sample_size=self.config.data.val_sample_size,
            apply_chat_template=self.config.data.apply_chat_template,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1, (
            "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."
        )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(
        self, inputs, outputs, scores, data_source=None, table_name=None
    ):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        # if data source is not None, prepare generations for each data source
        if data_source is not None:
            for source in set(data_source):
                source_index = [i for i, x in enumerate(data_source) if x == source]
                source_inputs = [inputs[i] for i in source_index]
                source_outputs = [outputs[i] for i in source_index]
                source_scores = [scores[i] for i in source_index]
                self._maybe_log_val_generations(
                    source_inputs,
                    source_outputs,
                    source_scores,
                    table_name=source.split(".")[0],
                )
            return

        # compute avg@n scores
        # val_n is extended with "repeat_interleave", i.e., [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
        # so we can reshape the scores to (num_samples, n) and compute the mean along the second axis
        avg_scores = (
            np.array(scores)
            .reshape(-1, self.config.actor_rollout_ref.rollout.val_kwargs.n)
            .mean(axis=1)
        )
        # not logging duplicate inputs
        unique_indexes = list(
            range(0, len(inputs), self.config.actor_rollout_ref.rollout.val_kwargs.n)
        )

        if self.config.trainer.output_acc_to_file:
            file_name = table_name.replace("/", "-")
            with open(f"{file_name}_acc.txt", "w") as f:
                for i, index in enumerate(unique_indexes):
                    question = (
                        inputs[index]
                        .split("User Question:")[-1]
                        .split("Assistant:")[0]
                        .strip()
                    )
                    f.write(f"problem {i + 1}: {question}\n")
                    f.write(f"avg score: {avg_scores[i]}\n\n")

        # prepare for true and false indexes
        avg_scores = np.repeat(avg_scores, 2)

        # save one correct sample and one incorrect sample
        true_false_indexes = []
        for i in unique_indexes:
            indexes_with_same_input = [
                j
                for j in range(
                    i, i + self.config.actor_rollout_ref.rollout.val_kwargs.n
                )
            ]
            scores_with_same_input = [scores[j] for j in indexes_with_same_input]
            true_false_indexes.append(
                indexes_with_same_input[np.argmax(scores_with_same_input)]
            )
            true_false_indexes.append(
                indexes_with_same_input[np.argmin(scores_with_same_input)]
            )

        inputs = [inputs[i] for i in true_false_indexes]
        outputs = [outputs[i] for i in true_false_indexes]
        output_scores = [scores[i] for i in true_false_indexes]
        assert len(avg_scores) == len(inputs) == len(outputs), (
            f"inconsistent length of scores, inputs, and outputs: {len(avg_scores)}, {len(inputs)}, {len(outputs)}"
        )

        if self.config.trainer.output_acc_to_file:
            # get current time
            import time

            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            # create a folder with current time
            folder_name = f"{file_name}/{current_time}"
            os.makedirs(folder_name, exist_ok=True)
            # we have both good and bad samples in inputs/outputs. So len(inputs) = 2*number_of_questions
            for i in range(len(inputs) // 2):
                with open(f"{folder_name}/problem-{i}-good.txt", "w") as f:
                    f.write(f"Score: {output_scores[i * 2]}\n\n")
                    f.write(f"{outputs[i * 2]}\n\n")
                with open(f"{folder_name}/problem-{i}-bad.txt", "w") as f:
                    f.write(f"Score: {output_scores[i * 2 + 1]}\n\n")
                    f.write(f"{outputs[i * 2 + 1]}\n\n")
            print(f"Accuracy results saved to {table_name}")

        samples = list(zip(inputs, outputs, output_scores, avg_scores))

        # We no longer sort and shuffle here since the prompt in each inputs may be different
        # samples.sort(key=lambda x: x[0])  # Sort by input text
        # # Use fixed random seed for deterministic shuffling
        # rng = np.random.RandomState(42)
        # rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        if table_name is not None:
            self.validation_generations_logger[table_name].log(
                self.config.trainer.logger, samples, self.global_steps
            )
        else:
            self.validation_generations_logger.log(
                self.config.trainer.logger, samples, self.global_steps
            )

    def compute_pass_at_k(self, results: list[list[bool]], k: int):
        """
        Compute the average pass@k metric for a list of problem results.

        Args:
            results: A list of lists of booleans, where each sublist represents the success of samples for a problem.
            k: The number of samples to consider (k in pass@k).

        Returns:
            The average pass@k score across all problems.
        """

        if k < 1:
            raise ValueError("k must be at least 1")

        pass_rates = []
        for problem in results:
            n = len(problem)
            if n < k:
                raise ValueError(
                    f"Each problem must have at least {k} samples, found {n}"
                )

            correct = sum(problem)
            if correct == 0:
                pass_rates.append(0.0)
                continue

            # Calculate the probability of failing all k trials
            fail_prob = 1.0
            for i in range(k):
                fail_prob *= (n - correct - i) / (n - i)

            pass_rates.append(1 - fail_prob)

        return sum(pass_rates) / len(pass_rates)

    def _validate(self):
        reward_tensor_lst = []
        reward_extra_info_dict: Dict[str, list[list[float]]] = (
            None  # the values are of shape (num_of_batch, batch_size)
        )
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_data_source = []

        if self.config.agent.tool_use:
            # Agent config preparation
            turn_predictor_config = None
            if hasattr(self.config.agent, 'adaptive_turns'):
                turn_predictor_config = TurnPredictorConfig(**self.config.agent.adaptive_turns)

            gen_config = GenerationConfig(
                max_turns=self.config.agent.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node
                * self.config.trainer.nnodes,
                rollout_n=1,
                mask_void_turns=False,  # no void turn masking during validation
                append_final_answer_func=self.config.agent.append_final_answer_func,
                turn_predictor_config=turn_predictor_config,
            )
            generation_manager = AgentHelper(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
            )

        print("=" * 20 + "Validation starts" + "=" * 20)
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_data_source.extend(test_batch.non_tensor_batch["data_source"])

            test_gen_batch = test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"]
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )

            if self.config.agent.tool_use:
                first_input_ids = (
                    test_gen_batch_padded.batch["input_ids"][
                        :, -gen_config.max_start_length :
                    ]
                    .clone()
                    .long()
                )
                test_output_gen_batch_padded = generation_manager.run_llm_loop(
                    gen_batch=test_gen_batch_padded,
                    initial_input_ids=first_input_ids,
                )
            else:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded
                )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_result = self.val_reward_fn(test_batch)

            # Handle both scalar and dictionary returns
            if isinstance(reward_result, dict):
                reward_tensor = reward_result["reward_tensor"]
                cur_data_source = test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
                if "extra_info" in reward_result:
                    if reward_extra_info_dict is None:
                        reward_extra_info_dict = {}
                        for key, extra_reward in reward_result["extra_info"].items():
                            for i, data_source in enumerate(cur_data_source):
                                composed_key = f"{key}_{data_source}"
                                if composed_key not in reward_extra_info_dict:
                                    reward_extra_info_dict[composed_key] = []
                                reward_extra_info_dict[composed_key].append(
                                    extra_reward[i]
                                )
            else:
                reward_tensor = reward_result
                cur_data_source = test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(cur_data_source)

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            data_source=sample_data_source,
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            assert (
                len(rewards) % self.config.actor_rollout_ref.rollout.val_kwargs.n == 0
            )
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)
            print(
                f"""Calculating pass@k rate for {data_source} with k={self.config.actor_rollout_ref.rollout.val_kwargs.k}"""
            )
            reward_per_test_sample = np.reshape(
                rewards, (-1, self.config.actor_rollout_ref.rollout.val_kwargs.n)
            )  # [N, n_val]
            pass_at_k_rate = self.compute_pass_at_k(
                reward_per_test_sample,
                k=self.config.actor_rollout_ref.rollout.val_kwargs.k,
            )
            print(f"[{data_source}]pass_at_k_rate:", pass_at_k_rate)
            metric_dict[
                f"val/test_score/{data_source}_pass@{self.config.actor_rollout_ref.rollout.val_kwargs.k}"
            ] = pass_at_k_rate

        if reward_extra_info_dict is not None:
            for key, extra_info_dict in reward_extra_info_dict.items():
                metric_dict[f"val/test_score_extra/{key}"] = np.mean(extra_info_dict)

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        if self.config.agent.tool_use:
            # Agent config preparation
            turn_predictor_config = None
            if hasattr(self.config.agent, 'adaptive_turns'):
                turn_predictor_config = TurnPredictorConfig(**self.config.agent.adaptive_turns)

            gen_config = GenerationConfig(
                max_turns=self.config.agent.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node
                * self.config.trainer.nnodes,
                rollout_n=self.config.actor_rollout_ref.rollout.n,
                mask_void_turns=self.config.actor_rollout_ref.actor.mask_void_turns,
                append_final_answer_func=self.config.agent.append_final_answer_func,
                turn_predictor_config=turn_predictor_config,
            )
            generation_manager = AgentHelper(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    if self.config.agent.tool_use:
                        # Get the first input_ids for generation from DataProto
                        first_input_ids = (
                            gen_batch.batch["input_ids"][
                                :, -gen_config.max_start_length :
                            ]
                            .clone()
                            .long()
                        )

                        with _timer("gen", timing_raw):
                            generation_manager.timing_raw = timing_raw
                            gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )
                    else:
                        # generate a batch
                        with _timer("gen", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch
                            )

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    response_info = _compute_response_info(batch)
                    response_lengths = response_info["response_length"]
                    # use the offical max response length to compute the masks
                    max_response_length = batch.batch["responses"].shape[-1]
                    cur_response_length = int(torch.max(response_lengths))

                    # generate example-level mask (True is masking)
                    if cur_response_length >= max_response_length:
                        sample_mask = (
                            response_lengths >= max_response_length
                        )  # [batch_size]
                        # if all samples are masked
                        if sample_mask.all():
                            print("All samples are masked, skip this batch.")
                            continue

                        adjusted_attention_mask = batch.batch["attention_mask"].clone()
                        for i, mask in enumerate(sample_mask):
                            if mask:
                                adjusted_attention_mask[i, -max_response_length:] = (
                                    0  # set response mask to 0
                                )

                        if self.config.trainer.remove_clip:
                            batch.batch["attention_mask"] = adjusted_attention_mask

                    if self.config.trainer.remove_clip:
                        # recompute response mask
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # create tool loss mask
                    if self.config.agent.tool_use and (
                        self.config.actor_rollout_ref.actor.mask_tool_output
                        or self.config.actor_rollout_ref.actor.mask_void_turns
                    ):
                        response_length = batch.batch["responses"].shape[-1]
                        response_mask = batch.batch["response_mask"][
                            :, -response_length:
                        ]

                        loss_mask = batch.batch["info_mask"][:, -response_length:]
                        if self.config.actor_rollout_ref.actor.mask_tool_output:
                            batch.batch["loss_mask"] = loss_mask * response_mask
                        else:
                            batch.batch["loss_mask"] = response_mask

                        if self.config.actor_rollout_ref.actor.mask_void_turns:
                            batch.batch["loss_mask"] = batch.batch[
                                "loss_mask"
                            ] * batch.batch["void_turn_mask"].reshape(-1, 1)

                        metrics.update(
                            {
                                "env/response_length": loss_mask.sum(axis=1)
                                .float()
                                .mean()
                                .item(),
                                "env/response_coverage": (
                                    loss_mask.sum() / response_mask.sum()
                                ).item(),
                            }
                        )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        with _timer("reward_fn", timing_raw):
                            # we combine with rule-based rm
                            reward_result = self.reward_fn(batch)

                        extra_rewards_info = None
                        if isinstance(reward_result, dict):
                            batch.batch["token_level_scores"] = reward_result[
                                "reward_tensor"
                            ]
                            if "extra_info" in reward_result:
                                extra_rewards_info = reward_result["extra_info"]
                        else:
                            batch.batch["token_level_scores"] = reward_result

                        reward_tensor = batch.batch["token_level_scores"]

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        solve_high = 0
                        solve_low = 0
                        over_long_all = 0
                        filter_all = 0
                        op_clip_num = 0
                        op_token_clip_num = 0
                        repeatness_clip_num = 0
                        number_larger_than_max_all = 0
                        min_rollout_n = self.config.actor_rollout_ref.rollout.min_n
                        group_length_vars = []

                        with _timer("rejection_sample", timing_raw):
                            for uid in unique_uids:
                                is_filtered = False
                                uid_mask = uids == uid

                                # Check if all rewards are 0 or all are 1 for this uid
                                if self.config.trainer.rejection_sample:
                                    with _timer("accuracy_clip", timing_raw):
                                        uid_rewards = reward_tensor[uid_mask].sum(
                                            -1
                                        )  # Sum rewards for each sequence
                                        # mask groups if its variance is too small
                                        if uid_rewards.std().item() < 1e-3:
                                            valid_mask[uid_mask] = False
                                            is_filtered = True
                                        elif self.config.trainer.acc_filter:
                                            acc_high = (
                                                self.config.trainer.acc_filter_high
                                            )
                                            acc_low = self.config.trainer.acc_filter_low
                                            if (
                                                uid_rewards.mean().item() < acc_low
                                                or uid_rewards.mean().item() > acc_high
                                            ):
                                                valid_mask[uid_mask] = False
                                                is_filtered = True
                                                if uid_rewards.mean().item() < acc_low:
                                                    solve_low += 1
                                                else:
                                                    solve_high += 1

                                    if (uid_rewards == 0).all():
                                        solve_none += 1
                                    # (WARNING) this should be tuned if we incorporate K1.5 length reward
                                    if (uid_rewards == 1).all():
                                        solve_all += 1

                                # get response lengths of uid
                                with _timer("rs_post_process", timing_raw):
                                    uid_response_lenghts = response_lengths[uid_mask]
                                    # compute mean and std of response lengths
                                    length_std = uid_response_lenghts.std().item()
                                    group_length_vars.append(length_std)

                                    # check how many response lengths are larger than max_response_length
                                    num_larger_than_max = (
                                        (uid_response_lenghts >= max_response_length)
                                        .sum()
                                        .item()
                                    )
                                    number_larger_than_max_all += num_larger_than_max
                                    # if all response lengths are larger than max_response_length, set mask to False
                                    if num_larger_than_max == len(uid_response_lenghts):
                                        valid_mask[uid_mask] = False
                                        is_filtered = True
                                        over_long_all += 1
                                    elif self.config.trainer.remove_clip:
                                        # if the mean of response lengths is larger than max_response_length, set mask to False
                                        if (
                                            len(uid_response_lenghts)
                                            - num_larger_than_max
                                            < min_rollout_n
                                        ):
                                            valid_mask[uid_mask] = False
                                            is_filtered = True
                                            over_long_all += 1

                                    if is_filtered:
                                        filter_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_high"] = solve_high
                        metrics["batch/solve_low"] = solve_low
                        metrics["batch/old_prob_clip_number"] = op_clip_num
                        metrics["batch/old_prob_token_clip_number"] = op_token_clip_num
                        metrics["batch/repeatness_clip_number"] = repeatness_clip_num
                        metrics["batch/over_long_all"] = over_long_all
                        metrics["batch/response_clip_number"] = (
                            number_larger_than_max_all
                        )
                        metrics["batch/length_vars"] = np.mean(group_length_vars)
                        metrics["batch/filter_all"] = filter_all

                        if (
                            self.config.trainer.rejection_sample
                            or self.config.trainer.remove_clip
                        ):
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                print("No valid samples remain, skip this batch")
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]
                            batch = dataprotoitem_to_dataproto(batch)

                            valid_query_size = (
                                batch.batch["input_ids"].shape[0]
                                // self.config.actor_rollout_ref.rollout.n
                            )
                            metrics["batch/effective_batch_size"] = min(
                                valid_query_size, self.config.data.train_batch_size
                            )

                            if valid_query_size < self.config.data.train_batch_size:
                                # calculate the times of copy
                                over_replicas = (
                                    1
                                    + (self.config.data.train_batch_size - 1)
                                    // valid_query_size
                                )
                                batch = batch.repeat_with_uid_suffix(over_replicas)

                            # if valid_query_size is equal to train_batch_size
                            expected_input_ids_size = (
                                self.config.actor_rollout_ref.rollout.n
                                * self.config.data.train_batch_size
                            )
                            if (
                                batch.batch["input_ids"].shape[0]
                                == expected_input_ids_size
                            ):
                                # do nothing, because we have already sampled enough samples
                                pass
                            elif (
                                batch.batch["input_ids"].shape[0]
                                < expected_input_ids_size
                            ):
                                raise ValueError(
                                    f"valid_query_size is less than the required batch size, which is not expected."
                                )
                            else:
                                # if valid_query_size is larger than the required batch size, it is usually because we have upsampled the training data. We should first decide to keep which query
                                valid_query_size = (
                                    batch.batch["input_ids"].shape[0]
                                    // self.config.actor_rollout_ref.rollout.n
                                )
                                query_size_mask = torch.zeros(
                                    valid_query_size, dtype=torch.bool
                                )
                                indices = torch.randperm(valid_query_size)
                                # randomly sample valid_query_size samples
                                query_size_mask[
                                    indices[: self.config.data.train_batch_size]
                                ] = True
                                size_mask = torch.zeros(
                                    batch.batch["input_ids"].shape[0], dtype=torch.bool
                                )
                                # use query size mask to select the size
                                sel_uids = batch.non_tensor_batch["uid"]
                                sel_unique_uids = np.unique(sel_uids)
                                for idx, sel_uid in enumerate(sel_unique_uids):
                                    sel_uid_mask = sel_uids == sel_uid
                                    if query_size_mask[idx]:
                                        size_mask[sel_uid_mask] = True
                                batch = batch[size_mask]
                                batch = dataprotoitem_to_dataproto(batch)

                            assert (
                                batch.batch["input_ids"].shape[0]
                                == expected_input_ids_size
                            ), (
                                f"batch size is not equal to train_batch_size, which is {batch.batch['input_ids'].shape[0]} vs {expected_input_ids_size}"
                            )
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                                mask_tool_output=self.config.actor_rollout_ref.actor.mask_tool_output
                                and self.config.agent.tool_use,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                # Add extra rewards metrics if they exist
                if extra_rewards_info is not None:
                    for key, sequence_extra in extra_rewards_info.items():
                        metrics.update(
                            {
                                f"critic/rewards_extra/{key}/mean": np.mean(
                                    sequence_extra
                                ),
                                f"critic/rewards_extra/{key}/max": np.max(
                                    sequence_extra
                                ),
                                f"critic/rewards_extra/{key}/min": np.min(
                                    sequence_extra
                                ),
                                f"critic/rewards_extra/{key}/var": np.var(
                                    sequence_extra
                                ),
                            }
                        )

                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
