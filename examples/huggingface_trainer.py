#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
HuggingFace Trainer with Differential Privacy support.
This example demonstrates an integration of Opacus with HuggingFace Trainer for training models with DP-SGD.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import datasets
import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader

from opacus.grad_sample.utils import wrap_model
from opacus.optimizers import AdaClipDPOptimizer, DPOptimizer, get_optimizer_class
from opacus.utils.batch_memory_manager import wrap_data_loader
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
    logging,
)
from transformers.trainer_callback import ExportableState


logger = logging.get_logger(__name__)


@dataclass
class PrivacyArguments:
    """
    Arguments for differentially private training.
    """

    accountant: str = "rdp"
    grad_sample_mode: str = "hooks"
    per_sample_max_grad_norm: float = 0.5
    clipping: str = "flat"
    poisson_sampling: bool = True
    min_clipbound: float = 0.05
    max_clipbound: float = 1e8
    clipbound_learning_rate: float = 0.2
    target_unclipped_quantile: float = 0.5
    unclipped_num_std: float = 1.0
    noise_multiplier: Optional[float] = None
    target_epsilon: Optional[float] = None
    target_delta: Optional[float] = None

    def precalculate(self, num_samples: int, sample_rate: float, steps: int):
        """
        Precalculate noise multiplier if not provided.
        """
        if self.target_delta is None:
            self.target_delta = 1.0 / num_samples

        if self.noise_multiplier is not None:
            return

        if self.target_epsilon is not None:
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                sample_rate=sample_rate,
                steps=steps,
                accountant=self.accountant,
            )
        else:
            raise ValueError(
                "Either noise_multiplier or target_epsilon must be specified."
            )


class DPCallback(TrainerCallback, ExportableState):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with Opacus.
    """

    def __init__(
        self,
        accountant: str,
        gradient_accumulation_steps: int,
        target_delta: float,
        max_epsilon: float = None,
    ) -> None:
        self.accountant = create_accountant(accountant)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.target_delta = target_delta
        self.max_epsilon = max_epsilon

    def get_optimizer_callback(self, sample_rate):
        return self.accountant.get_optimizer_hook_fn(sample_rate)

    def on_train_begin(self, args, state, control, **kwargs):
        return self._check_max_privacy_budget_exceeded(control)

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        optimizer = self._get_dp_optimizer(optimizer)

        # trainer samples one extra element at the beginning of each epoch, cleaning it up if present
        while len(optimizer._step_skip_queue) > self.gradient_accumulation_steps:
            optimizer._step_skip_queue.pop(0)

    def on_substep_end(self, args, state, control, optimizer=None, **kwargs):
        optimizer = self._get_dp_optimizer(optimizer)

        # gradients should be cleared after each substep with poisson sampling
        # precalculated grad_sample will stay until the final aggregation
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        optimizer = self._get_dp_optimizer(optimizer)

        # gradients should be cleared after each substep with poisson sampling
        # precalculated grad_sample will stay until the final aggregation
        # optimizer.step() is executed by the trainer
        optimizer.zero_grad(set_to_none=True)

    def on_evaluate(self, args, state, control, optimizer=None, metrics=None, **kwargs):
        return self._check_max_privacy_budget_exceeded(control)

    def get_privacy_metrics(self):
        metrics = {}
        if self.target_delta is not None:
            metrics["privacy_epsilon"] = (
                self.accountant.get_epsilon(self.target_delta)
                if len(self.accountant.history) > 0
                else 0.0
            )

        return metrics

    def _get_dp_optimizer(self, optimizer) -> DPOptimizer:
        for _ in range(10):
            if isinstance(optimizer, DPOptimizer):
                return optimizer
            elif hasattr(optimizer, "optimizer"):  # accelerate.Optimizer
                optimizer = optimizer.optimizer
            elif hasattr(optimizer, "_optimizer"):
                optimizer = optimizer._optimizer
            else:
                break

        raise ValueError(f"Expected DPOptimizer, got {type(optimizer)}")

    def _check_max_privacy_budget_exceeded(
        self, control: TrainerControl
    ) -> TrainerControl:
        metrics = self.get_privacy_metrics()
        if (
            "privacy_epsilon" in metrics
            and self.max_epsilon is not None
            and metrics["privacy_epsilon"] >= self.max_epsilon
        ):
            logger.warning(
                f"Max epsilon exceeded: {metrics['privacy_epsilon']} >= {self.max_epsilon}."
                "Stopping training..."
            )
            control.should_training_stop = True

        return control

    @property
    def _accountant_state_dict(self):
        return self.accountant.state_dict()

    @_accountant_state_dict.setter
    def _accountant_state_dict(self, state_dict):
        self.accountant.load_state_dict(state_dict)

    def state(self) -> dict:
        return {
            "args": {
                "accountant": self.accountant.mechanism(),
                "target_delta": self.target_delta,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_epsilon": self.max_epsilon,
            },
            "attributes": {
                "_accountant_state_dict": self._accountant_state_dict,
            },
        }


class DPTrainer(Trainer):
    def __init__(
        self,
        model: Union[nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Union[datasets.Dataset, torch.utils.data.Dataset] = None,
        privacy_args: PrivacyArguments = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        **kwargs,
    ):
        """Huggingface Trainer with Differential Privacy support.

        Args:
            model: Model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            privacy_args: Privacy arguments for differential private training.
            compute_metrics: Custom evaluation metrics.
            callbacks: Training callbacks.
            kwargs: Additional keyword arguments passed to Trainer.
        """
        self.privacy_args = privacy_args
        if not self.privacy_args:
            raise ValueError("Privacy arguments must be provided.")

        dataset_size = len(train_dataset)
        if (
            isinstance(
                train_dataset,
                (datasets.IterableDataset, torch.utils.data.IterableDataset),
            )
            and privacy_args.poisson_sampling
        ):
            raise ValueError(
                "IterableDataset is not supported by DPTrainer when poisson_sampling is True."
            )

        if (
            args.save_strategy
            and args.save_steps
            and not args.restore_callback_states_from_checkpoint
        ):
            warnings.warn(
                "Save strategy is set but restore_callback_states_from_checkpoint is false. "
                "Accountant states will not be restored from the checkpoint leading to the incorrect "
                "privacy budget estimates. Setting restore_callback_states_from_checkpoint to True"
            )
            args.restore_callback_states_from_checkpoint = True

        sample_rate = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            / dataset_size
        )

        self.privacy_args.precalculate(
            num_samples=dataset_size,
            sample_rate=sample_rate,
            steps=(
                args.max_steps // args.gradient_accumulation_steps
                if args.max_steps and args.max_steps != -1
                else math.ceil(1 / sample_rate) * args.num_train_epochs
            ),
        )

        logger.info(
            f"Using privacy noise multiplier: {self.privacy_args.noise_multiplier}"
        )

        # Attach hooks using wrap_model (non-wrapping mode)
        self.hooks = wrap_model(
            model,
            grad_sample_mode=self.privacy_args.grad_sample_mode,
            wrap_model=False,
        )

        dp_callback = DPCallback(
            accountant=self.privacy_args.accountant,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            target_delta=self.privacy_args.target_delta,
            max_epsilon=self.privacy_args.target_epsilon,
        )
        callbacks = callbacks or []
        callbacks.append(dp_callback)

        def compute_privacy_metrics(*args, **kwargs):
            if compute_metrics:
                metrics = compute_metrics(*args, **kwargs)
            else:
                metrics = {}

            privacy_metrics = dp_callback.get_privacy_metrics()
            metrics.update(privacy_metrics)

            return metrics

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            callbacks=callbacks,
            compute_metrics=compute_privacy_metrics,
            **kwargs,
        )

        optimizer = self.create_optimizer()

        if "ghost" in self.privacy_args.grad_sample_mode:
            # We assume the model has a loss_function attribute or use default
            base_criterion = getattr(model, "loss_function", nn.CrossEntropyLoss())

            criterion = DPLossFastGradientClipping(
                self.hooks, optimizer, base_criterion, "mean"
            )
            self.set_loss_function_recursively(model, criterion)

        optimizer.attach_step_hook(
            dp_callback.get_optimizer_callback(sample_rate=sample_rate)
        )

    def create_optimizer(self):
        if self.optimizer:
            return self.optimizer

        self.optimizer = super().create_optimizer()

        optim_class = get_optimizer_class(
            clipping=self.privacy_args.clipping,
            distributed=False,
            grad_sample_mode=self.privacy_args.grad_sample_mode,
        )

        kwargs = {
            "optimizer": self.optimizer,
            "noise_multiplier": self.privacy_args.noise_multiplier,
            "expected_batch_size": self.args.per_device_train_batch_size,
            "max_grad_norm": self.privacy_args.per_sample_max_grad_norm,
            "loss_reduction": "mean",
        }

        if issubclass(optim_class, AdaClipDPOptimizer):
            kwargs.update(
                {
                    "max_clipbound": self.privacy_args.max_clipbound,
                    "min_clipbound": self.privacy_args.min_clipbound,
                    "clipbound_learning_rate": self.privacy_args.clipbound_learning_rate,
                    "target_unclipped_quantile": self.privacy_args.target_unclipped_quantile,
                    "unclipped_num_std": self.privacy_args.unclipped_num_std,
                }
            )

        self.optimizer = optim_class(**kwargs)

        return self.optimizer

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        data_loader = self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

        if self.privacy_args.poisson_sampling:
            data_loader = DPDataLoader.from_data_loader(data_loader)

        data_loader = wrap_data_loader(
            data_loader=data_loader,
            optimizer=self.create_optimizer(),
            max_batch_size=self._train_batch_size,
        )

        return data_loader

    @staticmethod
    def set_loss_function_recursively(model, new_loss_function, max_depth=10):
        """
        Recursively set the loss_function property on all models in the wrapper hierarchy.
        """
        visited = set()

        def _set_recursive(current_model, depth=0):
            if depth >= max_depth:
                raise ValueError(f"Maximum unwrapping depth {max_depth} reached")

            model_id = id(current_model)
            if model_id in visited:
                return
            visited.add(model_id)

            # Set loss_function if it exists on the current model
            if hasattr(current_model, "loss_function"):
                current_model.loss_function = new_loss_function

            # Continue recursively through wrapper attributes
            wrapper_attrs = [
                "_module",  # GradSampleModule
                "base_model",  # PeftModelForCausalLM, PEFT models
                "model",  # LoraModel
                "module",  # DistributedDataParallel, DataParallel
            ]

            for attr_name in wrapper_attrs:
                if hasattr(current_model, attr_name):
                    wrapped_model = getattr(current_model, attr_name)
                    # Make sure it's actually a model object and not None
                    if wrapped_model is not None and hasattr(
                        wrapped_model, "__class__"
                    ):
                        try:
                            _set_recursive(wrapped_model, depth + 1)
                        except (ValueError, AttributeError):
                            continue

        _set_recursive(model)

    def detach_model(self) -> nn.Module:
        """Detach the model from the hooks and return the model."""
        self.hooks.cleanup()
        return self.model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Opacus HuggingFace Trainer Example")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name from HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Dataset name from HuggingFace Hub",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Accumulation steps"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--target_epsilon", type=float, default=9.0, help="Target epsilon"
    )
    parser.add_argument(
        "--no_lora", action="store_true", help="Disable LoRA", default=False
    )
    parser.add_argument(
        "--grad_sample_mode",
        type=str,
        default="hooks",
        help="Opacus grad_sample_mode",
    )
    args = parser.parse_args()

    # 1. Load dataset
    raw_datasets = datasets.load_dataset(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Assuming only 'train' split is provided, we split it for training and evaluation
    dataset = raw_datasets["train"].shuffle(seed=42).rename_column("label", "labels")
    small_train_dataset = dataset.take(500).map(tokenize_function, batched=True)
    small_train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    small_eval_dataset = (
        dataset.skip(500).take(100).map(tokenize_function, batched=True)
    )
    small_eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 2. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    if not args.no_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
        )
        model = get_peft_model(model, peft_config)
        print("LoRA applied to the model.")

    # 3. Define training arguments
    training_args = TrainingArguments(
        output_dir="test_trainer",
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=1,
        restore_callback_states_from_checkpoint=True,
        report_to=None,
    )

    # 4. Initialize PrivacyArguments
    privacy_args = PrivacyArguments(
        target_epsilon=args.target_epsilon,
        target_delta=1e-5,
        grad_sample_mode=args.grad_sample_mode,
    )

    # 5. Initialize DPTrainer
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        privacy_args=privacy_args,
    )

    # 6. Train
    print("Starting training with DPTrainer...")
    trainer.train()

    # 7. Final epsilon
    # Epsilon is also available in metrics due to compute_privacy_metrics
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    # 8. Cleanup
    trainer.detach_model()


if __name__ == "__main__":
    main()
