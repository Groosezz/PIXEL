from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, is_torch_tpu_available
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, nested_concat, nested_numpify
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import logging

from .utils.optimization import get_cosine_schedule_to_min_lr_with_warmup
from .utils.training import debug_log_inputs

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class PIXELTrainer(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Uncomment this to visualize inputs
        # debug_log_inputs(inputs)

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class PIXELTrainerForPretraining(PIXELTrainer):
    """
    PIXELTrainer for pretraining
    """

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_to_min_lr_with_warmup(
                self.optimizer if optimizer is None else optimizer,
                self.args.get_warmup_steps(num_training_steps),
                num_training_steps,
                self.args.learning_rate,
            )
        return self.lr_scheduler

class TrainerForBARTPretraining(Trainer):
    """
    PIXELTrainer for pretraining
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        """

        # Uncomment this to visualize inputs
        # debug_log_inputs(inputs)

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        #if labels is not None:
            #loss = self.label_smoother(outputs, labels)
        #else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_to_min_lr_with_warmup(
                self.optimizer if optimizer is None else optimizer,
                self.args.get_warmup_steps(num_training_steps),
                num_training_steps,
                self.args.learning_rate,
            )
        return self.lr_scheduler


class BartTrainer(Trainer):
    """
    PIXELTrainer for pretraining
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        """

        # Uncomment this to visualize inputs
        # debug_log_inputs(inputs)

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        #if labels is not None:
            #loss = self.label_smoother(outputs, labels)
        #else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

class PIXELTrainerForQuestionAnswering(PIXELTrainer):
    """
    PixelTrainer for extractive question answering
    """

    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(eval_dataloader, description="Evaluation")
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(predict_dataloader, description="Prediction")
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
