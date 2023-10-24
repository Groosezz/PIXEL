#!/usr/bin/env pythonsoftware
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
import os
import sys


from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import submitit

import datasets
import torch
from torch import nn
import transformers
from datasets import interleave_datasets, load_dataset, load_from_disk
from pixel import (
    PIXELConfig,
    PIXELEmbeddings,
    BartForConditionalGeneration,
    TrainerForBARTPretraining,
    SpanMaskingGenerator,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    get_2d_sincos_pos_embed
)
from transformers import HfArgumentParser, TrainingArguments, ViTFeatureExtractor
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, Resize, ToTensor
""" Pre-training a PIXEL model as an MAE (masked autoencoder)"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_names: str = field(metadata={"help": "Path of train dataset in local disk"})
    train_splits: str = field(metadata={"help": "Name of the training dataset split."})
    validation_dataset_name: str = field(metadata={"help": "Path of validation dataset in local"})
    validation_split: str = field(metadata={"help": "Name of the validation dataset split."})
    dataset_caches: Optional[str] = field(default=None, metadata={"help": "Directory where the dataset is cached"})
    train_dataset_configs: str = field(default=None, metadata={"help": "Train dataset config/subset"})
    isBinaryPixel: Optional[bool] = field(default=False, metadata={"help": "whether or not to binarize the pixel values from gray style(0-256) to (-1,1)"})
    BinaryLoss: str = field(default='mse', metadata={"help": "can select from 'mse', 'focal, 'bce']"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    buffer_size: Optional[int] = field(default=1000, metadata={"help": "the buffer size used in shuffling the iteratble datasets"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "Whether to stream the training dataset"})
    do_normalize: Optional[bool] = field(
        default=False, metadata={"help": "Whether to normalize to model's feature extractor's mean and std."}
    )

    def __post_init__(self):
        self.train_dataset_names = self.train_dataset_names.split(",")
        self.train_splits = self.train_splits.split(",")
        if self.train_dataset_configs:
            self.train_dataset_configs = self.train_dataset_configs.split(",")
        else:
            self.train_dataset_configs = [None] * len(self.train_dataset_names)
        if self.dataset_caches:
            self.dataset_caches = self.dataset_caches.split(",")
        else:
            self.dataset_caches = [None] * len(self.train_dataset_names)
        assert (
            len(self.train_dataset_names)
            == len(self.train_splits)
            == len(self.train_dataset_configs)
            == len(self.dataset_caches)
        )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    text_renderer_name_or_path: str = field(
        metadata={
            "help": "Path / Huggingface identifier of the text renderer that was used to prerender the "
            "training/validation data."
        }
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: str = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )

    mask_ratio: float = field(
        default=0.25, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    span_masking: bool = field(
        default=False, metadata={"help": "Whether to use span masking instead of random masking."}
    )
    masking_max_span_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum span length that can be masked when using span masking."}
    )
    masking_spacing: Optional[int] = field(
        default=None,
        metadata={
            "help": "Spacing between masked spans. Defaults to the length of the span."
            "Use this argument to set it to a fixed number of patches."
            "Recommended setting: For masking ratio <= 0.4 leave the default"
            "For ratios between 0.4 and 0.7 set it to 1. For higher, set it to 0"
        },
    )
    masking_cumulative_span_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of cumulative probabilities of sampling a span of length n"
            "when using span masking. Must be a list of size model_args.masking_max_span_length."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks"}
    )
    decoder_layers: int = field(
        default=12, metadata={"help": "Number of decoder layers"}
    )
    decoder_ffn_dim: int = field(
        default=3072, metadata={"help": "Dimensionality of the intermediate (often named feed-forward) layer in decoder."}
    )
    vocab_size: int = field(
        default=192, metadata={"help": "patch_size**2*n_channels"}
    )

    def __post_init__(self):
        if self.masking_cumulative_span_weights is not None:
            self.masking_cumulative_span_weights = [float(w) for w in self.masking_cumulative_span_weights.split(",")] 

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1.5e-4, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )


def collate_fn(examples):
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    decoder_attention_mask = torch.stack([example["decoder_attention_mask"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_attention_mask": decoder_attention_mask, "labels": labels}
    if "patch_mask" in examples[0]:
        patch_mask = torch.stack([example["patch_mask"] for example in examples])
        inputs.update({"patch_mask": patch_mask})
    return inputs


def main(config_dict: Dict[str, Any] = None):
    # environment variables
    #os.environ["SUBMITIT_EXECUTOR"] = "slurm"
    #os.environ["TRANSFORMERS_OFFLINE"] = '1'
    #condaos.environ["SUBMITIT_EXECUTOR"] = '1'
    
    # CUDNN backends
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    #torch.distributed.init_process_group()
    

    job_env = submitit.JobEnvironment()
    os.environ["LOCAL_RANK"] = str(job_env.local_rank)
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    else:
        os.environ["RANK"] = str(job_env.global_rank)
    os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"WORLD_SIZE: {world_size}")
    print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if not config_dict:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_dict(config_dict)
    
    # set up in submit_training
    
    #training_args.output_dir = os.path.join(
            #training_args.output_dir.replace("%j", str(job_env.job_id)), "outputs")
    ### here we also return to the same output_dir
    
    training_args.run_name = training_args.run_name.replace("%j", str(job_env.job_id))
    

        
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # for test purpose
    #train_dataset = [ load_from_disk(d_name) for d_name in data_args.train_dataset_names][0]   
    
    train_datasets = [load_from_disk(d_name) for d_name in data_args.train_dataset_names]
    if len(data_args.train_dataset_names) > 1:
        dataset_sizes = [ds._info.splits.total_num_examples for ds in train_datasets]
        combined_size = sum(dataset_sizes)
        dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
        train_dataset = interleave_datasets(train_datasets, probabilities=dataset_sampling_probs, seed=training_args.seed)
        logger.info("***** Interleaving training datasets *****")
        for d_name, d_config, d_split, d_sampling_prob, d_cache in zip(
            data_args.train_dataset_names,
            data_args.train_dataset_configs,
            data_args.train_splits,
            dataset_sampling_probs,
            data_args.dataset_caches,
        ):
            logger.info(
                f"\tDataset name = {d_name}, config = {d_config}, split = {d_split}, "
                f"sampling probability = {d_sampling_prob:.3f}, cache = {d_cache}"
            )
    else: # for debug purpose
        train_dataset = train_datasets[0]
    
    validation_dataset = load_from_disk(data_args.validation_dataset_name)
    logger.info(f"The sizes of datasets are:" 
                f"train_datasets-{len(train_dataset)}, validation_dataset-{len(validation_dataset)}")


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if model_args.config_name:
        config = PIXELConfig.from_pretrained(
            model_args.config_name,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    elif model_args.model_name_or_path:
        config = PIXELConfig.from_pretrained(
            model_args.model_name_or_path,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    else:
        config = PIXELConfig(
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
            "decoder_num_hidden_layers": model_args.decoder_layers,
            "decoder_intermediate_size": model_args.decoder_ffn_dim,
            "vocab_size": model_args.vocab_size,
            "isBinary": data_args.isBinaryPixel,
            "BinaryLoss": data_args.BinaryLoss,
            "architectures": [BartForConditionalGeneration.__name__]
        }
    )

    # Create model
    if model_args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            **config_kwargs,
        )
    else:
        logger.info("Training new model from scratch")
        model = BartForConditionalGeneration(config)
     
    # Load text renderer
    text_renderer = PyGameTextRenderer.from_pretrained(model_args.text_renderer_name_or_path, **config_kwargs)

    # Load or create feature extractor
    if model_args.feature_extractor_name:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.feature_extractor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        feature_extractor = ViTFeatureExtractor()

    # Adjust image size for BART
    image_height = text_renderer.pixels_per_patch
    image_width = text_renderer.pixels_per_patch * text_renderer.max_seq_length
    model.config.image_size = (image_height, image_width)
    model.config.patch_size = image_height
    model.image_size = (image_height, image_width)
    feature_extractor.size = (image_height, image_width)

    # Reinitialize embeddings
    #if data_args.isPixelBinary:
    if data_args.isBinaryPixel:
        model.config.num_channels = 1
        model.config.vocab_size = model.config.patch_size **2*model.config.num_channels
        logger.info(f"Use Binary Pixel values, change the channel size from 3 to {model.config.num_channels}, change the vocab size from 192 to {config.vocab_size}")
        model.lm_head = nn.Linear(model.config.hidden_size, config.vocab_size, bias=False)
        model.model.vit.embeddings = PIXELEmbeddings(model.config)
    else:
        model.model.vit.embeddings = PIXELEmbeddings(model.config)
    
    # set all params to requre grads
    for param in model.parameters():
        param.requires_grad = True
        
    logger.info("***** Final model config *****")
    logger.info(config)

    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f"Total parameters count: {total_params}")
    encoder_params = sum([p.numel() for p in model.model.vit.parameters()])
    logger.info(f"Encoder parameters count: {encoder_params}")
    encoder_embedding_params = sum([p.numel() for p in model.model.vit.embeddings.parameters()])
    logger.info(f"Encoder embeddings parameters count: {encoder_embedding_params}")
    decoder_params = sum([p.numel() for p in model.model.gpt.parameters()])
    logger.info(f"Decoder parameters count: {decoder_params}")
    
    # Get patch mask generator if span masking
    if model_args.span_masking and model_args.masking_max_span_length and model_args.masking_cumulative_span_weights:
        logger.info(
            f'Applying span masking with "max_span_length = {model_args.masking_max_span_length}" '
            f', "cumulative_span_weights = {model_args.masking_cumulative_span_weights}" '
            f' and "spacing = {model_args.masking_spacing if model_args.masking_spacing else "span"}"'
        )
        patch_mask_generator = SpanMaskingGenerator(
            num_patches=text_renderer.max_seq_length,
            num_masking_patches=math.ceil(model_args.mask_ratio * text_renderer.max_seq_length),
            max_span_length=model_args.masking_max_span_length,
            spacing=model_args.masking_spacing if model_args.masking_spacing else "span",
            cumulative_span_weights=model_args.masking_cumulative_span_weights,
        )

    column_names = ["pixel_values", "text", "num_patches"]
    image_column_name = column_names[0]

    if data_args.do_normalize:
        image_mean = feature_extractor.image_mean
        image_std = feature_extractor.image_std
    else:
        image_mean, image_std = (None, None)
        feature_extractor.do_normalize = data_args.do_normalize

    # Set transformations --- resize by default and optionally apply normalization
    if data_args.isBinaryPixel:
        transforms = Compose([
            Resize(feature_extractor.size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            lambda x: x>=1,
            lambda x: x.float(),])
    else:
        image_mean, image_std = (None, None)
        feature_extractor.do_normalize = False

        # Set transformations --- resize by default and optionally apply normalization
        transforms = get_transforms(
            do_resize=True,
            size=(image_height, image_width),
            do_normalize=False,
            image_mean=image_mean,
            image_std=image_std,
        )

    logger.info(f"Applied transformations: {transforms}")
    
    def patchify(imgs):
        """
        imgs: (N, C, H, W) x: (N, L, patch_size**2 *C)
        """
        """
        imgs: (C, H, W) x: (L, patch_size**2 *C)
        """
        p = imgs.shape[1]
        c = model.config.num_channels
        assert imgs.shape[1] % p == 0 and imgs.shape[2] % p == 0

        h = imgs.shape[1] // p
        w = imgs.shape[2] // p
        x = imgs.reshape(shape=(c, h, p, w, p))
        x = torch.einsum("chpwq->hwpqc", x)
        x = x.reshape(shape=( h * w, p ** 2 * c))

        return x
    
    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        examples["input_ids"] = [transforms(image) for image in examples[image_column_name]]
        examples["attention_mask"] = [get_attention_mask(num_patches) for num_patches in examples["num_patches"]]
        examples["labels"] = [patchify(input_id) for input_id in examples['input_ids']]
        examples["decoder_attention_mask"] = examples["attention_mask"] 
        if model_args.span_masking:
            examples["patch_mask"] = [
                torch.tensor(patch_mask_generator(num_patches + 1), dtype=torch.float32)
                for num_patches in examples["num_patches"]
            ]
        return examples

    if training_args.do_train:
        if data_args.streaming:
            # turn the dataset into the iterable dataset and make the use of the speeding in the shuffling and mapping
            train_dataset = train_dataset.to_iterable_dataset()
            train_dataset = train_dataset.with_format("torch")
            train_dataset = train_dataset.shuffle(training_args.seed, buffer_size=data_args.buffer_size)
        # Filter out examples that are less than one row long in the squared input image/ for test purpose
        #train_dataset = train_dataset.filter(lambda x: (x["num_patches"] >= 22))
       # if data_args.max_train_samples is not None:
           # train_dataset = train_dataset.shuffle(seed=training_args.seed).select(
               # range(data_args.max_train_samples)
           # 
        # Set training transforms
        if data_args.streaming:
            train_dataset = train_dataset.map(preprocess_images, batched=True, batch_size=1000)
        else:
            train_dataset.set_transform(preprocess_images)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            validation_dataset = validation_dataset.shuffle(seed=training_args.seed).select(
                range(data_args.max_eval_samples)
            )
        # Set the validation transforms
        validation_dataset.set_transform(preprocess_images)
    
    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = TrainerForBARTPretraining(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        tokenizer=text_renderer,
        data_collator=collate_fn,
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # Also save feature extractor together with model and text renderer
        feature_extractor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": "wikipedia + bookcorpus",
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
