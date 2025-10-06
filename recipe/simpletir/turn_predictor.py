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

from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np


@dataclass
class TurnPredictorConfig:
    """Configuration for adaptive turn budget predictor."""

    enable_adaptive_turns: bool = False
    min_turns: int = 1
    max_turns: int = 50
    confidence_threshold: float = 0.85
    base_turns: int = 5

    # Feature weights for heuristic prediction
    prompt_length_weight: float = 0.3
    complexity_weight: float = 0.4
    progress_weight: float = 0.3

    # Early stopping parameters
    enable_early_stopping: bool = True
    early_stop_window: int = 2
    early_stop_improvement_threshold: float = 0.01


class AdaptiveTurnPredictor:
    """
    Meta-controller that predicts optimal turn budget for each problem
    based on problem complexity signals and execution progress.
    """

    def __init__(self, config: TurnPredictorConfig):
        self.config = config
        self.turn_history = []
        self.performance_history = []

    def predict_turn_budget(
        self,
        prompt_length: int,
        prompt_ids: torch.Tensor = None,
        extra_features: dict = None
    ) -> int:
        """
        Predict optimal number of turns for a given problem.

        Args:
            prompt_length: Length of the input prompt in tokens
            prompt_ids: Optional tensor of prompt token IDs for complexity analysis
            extra_features: Optional dictionary of additional features

        Returns:
            Predicted optimal number of turns (clamped to min_turns and max_turns)
        """
        if not self.config.enable_adaptive_turns:
            return self.config.base_turns

        # Compute complexity score
        complexity_score = self._estimate_complexity(
            prompt_length, prompt_ids, extra_features
        )

        # Map complexity to turn budget using sigmoid scaling
        normalized_complexity = (complexity_score - 0.5) * 2
        turn_scaling = torch.sigmoid(torch.tensor(normalized_complexity)).item()

        # Calculate predicted turns
        turn_range = self.config.max_turns - self.config.min_turns
        predicted_turns = int(
            self.config.min_turns + turn_scaling * turn_range
        )

        return max(self.config.min_turns, min(predicted_turns, self.config.max_turns))

    def should_early_stop(
        self,
        current_turn: int,
        code_execution_results: List[dict],
        response_confidence: float = None
    ) -> Tuple[bool, float]:
        """
        Determine if generation should stop early based on confidence signals.

        Args:
            current_turn: Current turn number
            code_execution_results: List of execution results from recent turns
            response_confidence: Optional confidence score from the model

        Returns:
            Tuple of (should_stop, confidence_score)
        """
        if not self.config.enable_early_stopping:
            return False, 0.0

        if current_turn < self.config.min_turns:
            return False, 0.0

        # Compute confidence from execution results
        confidence = self._compute_confidence(
            code_execution_results, response_confidence
        )

        # Check if confidence exceeds threshold
        if confidence >= self.config.confidence_threshold:
            return True, confidence

        # Check for convergence (no improvement over window)
        if len(code_execution_results) >= self.config.early_stop_window:
            recent_results = code_execution_results[-self.config.early_stop_window:]
            if self._is_converged(recent_results):
                return True, confidence

        return False, confidence

    def _estimate_complexity(
        self,
        prompt_length: int,
        prompt_ids: torch.Tensor = None,
        extra_features: dict = None
    ) -> float:
        """
        Estimate problem complexity using heuristic features.

        Returns:
            Complexity score in range [0, 1]
        """
        features = []
        weights = []

        # Feature 1: Prompt length (normalized)
        max_reasonable_length = 4096
        length_score = min(prompt_length / max_reasonable_length, 1.0)
        features.append(length_score)
        weights.append(self.config.prompt_length_weight)

        # Feature 2: Token complexity (if available)
        if prompt_ids is not None:
            complexity_score = self._analyze_token_complexity(prompt_ids)
            features.append(complexity_score)
            weights.append(self.config.complexity_weight)
        else:
            # Use length as proxy
            features.append(length_score)
            weights.append(self.config.complexity_weight)

        # Feature 3: Historical performance (if available)
        if len(self.turn_history) > 0:
            avg_turns = np.mean(self.turn_history[-100:])
            max_observed = max(self.turn_history[-100:])
            progress_score = avg_turns / max(max_observed, 1.0)
            features.append(progress_score)
            weights.append(self.config.progress_weight)
        else:
            features.append(0.5)
            weights.append(self.config.progress_weight)

        # Additional features from extra_features dict
        if extra_features:
            if 'data_source' in extra_features:
                # Different datasets may have different difficulty patterns
                source_complexity = self._get_source_complexity(
                    extra_features['data_source']
                )
                features.append(source_complexity)
                weights.append(0.1)

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Weighted average
        complexity = sum(f * w for f, w in zip(features, weights))

        return np.clip(complexity, 0.0, 1.0)

    def _analyze_token_complexity(self, prompt_ids: torch.Tensor) -> float:
        """
        Analyze token sequence complexity using entropy and uniqueness.

        Args:
            prompt_ids: Tensor of token IDs

        Returns:
            Complexity score in range [0, 1]
        """
        if prompt_ids.numel() == 0:
            return 0.5

        # Flatten and convert to numpy
        ids = prompt_ids.flatten().cpu().numpy()

        # Compute token uniqueness ratio
        unique_ratio = len(np.unique(ids)) / max(len(ids), 1)

        # Compute simple entropy estimate
        _, counts = np.unique(ids, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        normalized_entropy = entropy / max(max_entropy, 1e-10)

        # Combine metrics
        complexity = 0.6 * unique_ratio + 0.4 * normalized_entropy

        return float(np.clip(complexity, 0.0, 1.0))

    def _get_source_complexity(self, data_source: str) -> float:
        """
        Map data source to expected complexity.

        Args:
            data_source: Name of the dataset

        Returns:
            Complexity score in range [0, 1]
        """
        complexity_map = {
            'aime': 0.9,
            'aime24': 0.9,
            'aime25': 0.9,
            'olympiad': 0.95,
            'math500': 0.7,
            'gsm8k': 0.3,
            'amc': 0.6,
            'hmmt': 0.8,
        }

        # Case-insensitive matching
        source_lower = data_source.lower()
        for key, value in complexity_map.items():
            if key in source_lower:
                return value

        # Default to medium complexity
        return 0.5

    def _compute_confidence(
        self,
        code_execution_results: List[dict],
        response_confidence: float = None
    ) -> float:
        """
        Compute confidence score from execution history and model signals.

        Args:
            code_execution_results: Recent execution results
            response_confidence: Optional model-provided confidence

        Returns:
            Confidence score in range [0, 1]
        """
        if not code_execution_results:
            return 0.0

        confidences = []
        weights = []

        # Check last execution success
        last_result = code_execution_results[-1]
        if last_result.get('has_valid_code', False):
            if last_result.get('execution_success', False):
                confidences.append(0.8)
                weights.append(0.4)
            else:
                confidences.append(0.3)
                weights.append(0.4)
        else:
            confidences.append(0.1)
            weights.append(0.4)

        # Check for final answer presence
        if last_result.get('has_final_answer', False):
            confidences.append(0.9)
            weights.append(0.3)

        # Use model confidence if available
        if response_confidence is not None:
            confidences.append(response_confidence)
            weights.append(0.3)

        # Compute weighted average
        if not confidences:
            return 0.0

        weight_sum = sum(weights)
        if weight_sum == 0:
            return np.mean(confidences)

        weighted_conf = sum(c * w for c, w in zip(confidences, weights)) / weight_sum

        return float(np.clip(weighted_conf, 0.0, 1.0))

    def _is_converged(self, recent_results: List[dict]) -> bool:
        """
        Check if recent results show convergence (no improvement).

        Args:
            recent_results: List of recent execution results

        Returns:
            True if converged, False otherwise
        """
        if len(recent_results) < 2:
            return False

        # Extract success indicators
        successes = [
            r.get('execution_success', False) for r in recent_results
        ]

        # Check if all recent attempts succeeded
        if all(successes):
            return True

        # Check if all recent attempts failed
        if not any(successes):
            return True

        # Check for output stability
        outputs = [r.get('output', '') for r in recent_results]
        if len(set(outputs)) == 1 and outputs[0]:
            # Same non-empty output repeated
            return True

        return False

    def update_statistics(self, turns_used: int, success: bool):
        """
        Update predictor statistics with completed trajectory.

        Args:
            turns_used: Number of turns used
            success: Whether the trajectory succeeded
        """
        self.turn_history.append(turns_used)
        self.performance_history.append(1.0 if success else 0.0)

        # Keep only recent history
        max_history = 1000
        if len(self.turn_history) > max_history:
            self.turn_history = self.turn_history[-max_history:]
            self.performance_history = self.performance_history[-max_history:]

    def get_statistics(self) -> dict:
        """
        Get current predictor statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.turn_history:
            return {
                'avg_turns': self.config.base_turns,
                'success_rate': 0.0,
                'num_samples': 0
            }

        return {
            'avg_turns': np.mean(self.turn_history),
            'median_turns': np.median(self.turn_history),
            'max_turns': np.max(self.turn_history),
            'min_turns': np.min(self.turn_history),
            'success_rate': np.mean(self.performance_history),
            'num_samples': len(self.turn_history)
        }
