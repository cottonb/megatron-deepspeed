# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import random
from functools import partial
from typing import Dict, Union

import torch
import torch.utils.data
from datasets import concatenate_datasets
from datasets import load_from_disk
from torch import Tensor

import megatron.core.models
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.utils import print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.global_vars import get_tokenizer
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_batch_on_this_cp_rank
from megatron.utils import get_ltor_masks_and_position_ids

from dlckpt.python.common.log import default_logger as logger


def _get_tokenizer():
    args = get_args()
    if not args.hf_tokenizer:
        return get_tokenizer()
    
    assert args.tokenizer_type == 'NullTokenizer', '--hf-tokenizer only works with NullTokenizer'

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    return tokenizer


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    rope_base = getattr(args, 'rope_base', 10000.0)
    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts is None:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
            else:
                from megatron.core.models.gpt.gpt_layer_specs import (
                    gpt_layer_with_transformer_engine_spec_moe,
                )
                transformer_layer_spec = gpt_layer_with_transformer_engine_spec_moe
        model: Union[GPTModel, megatron.model.GPTModel] = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=int(rope_base),
        )
    else:
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
        # (NOTE) This is a patch for base-variant RoPE
        # Recompute the inv_freq of RotaryEmbedding w.r.t. the base value set in arguments.
        # Note that this hack needs to sync with the changes in `megatron/core/rotary_pos_embedding.py`.
        rope_dim = (
            args.hidden_size // args.num_attention_heads
            if args.kv_channels is None
            else args.kv_channels
        )
        model.language_model.rotary_pos_emb.inv_freq = 1.0 / (
            rope_base
            ** (
                torch.arange(
                    0, rope_dim, 2, dtype=torch.float32, device=torch.cuda.current_device()
                )
                / rope_dim
            )
        )
    
    total_parameter = sum([param.nelement() for param in model.parameters()])
    print_rank_0('*'*30)
    print_rank_0('Total parameters: {:.3f}M'.format(total_parameter/1e6))
    print_rank_0('*'*30)
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()
    seq_length = args.max_position_embeddings

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # Items and their type.
    keys = ['input_ids', 'position_ids']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    # print_rank_0(f'data.keys: {data.keys()}')
    
    if 'position_ids' not in data:
        pos_ids = (
            torch.arange(seq_length + 1, dtype=datatype, device=data['input_ids'].device)
            .unsqueeze(0)
            .expand_as(data['input_ids'])
        ).clone()
        data['position_ids'] = pos_ids
    
    # (TODO) Find out why data_iterator returns a list.
    if isinstance(data['input_ids'], list) and isinstance(data['position_ids'], list):
        data['input_ids'] = torch.stack(data['input_ids']).T
        data['position_ids'] = torch.stack(data['position_ids']).T

    input_ids = data['input_ids'].to(dtype=datatype)
    position_ids = data['position_ids'].to(dtype=datatype)

    # No batching here, workaround for num-workers=0
    if len(input_ids.shape) < 2:
        input_ids = input_ids.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)

    data = {"input_ids": input_ids, "position_ids": position_ids}

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()  # Preprocessed to target seq length + 1
    position_ids = data_b['position_ids'].long()[:, :-1]

    labels = tokens_[:, 1:].contiguous()[:, :seq_length]
    tokens = tokens_[:, :-1].contiguous()[:, :seq_length]

    # Get the masks and postition ids.
    if args.use_flash_attn:
        loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
        attention_mask = loss_mask.bool().contiguous()
    else:  # consume a lot of memory when running over long sequence
        tokenizer = _get_tokenizer()
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
        )

    loss_mask[tokens == -100] = 0.0
    batch = {
        'tokens': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    import time

    start = time.time()
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()
    # print_rank_0(f"Time to get batch: {time.time() - start}")

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    # print_rank_0(model)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def _get_dataset(data_paths):
    datasets = []
    for data_path in data_paths:
        print_rank_0(f"Loading data from {data_path}")
        dataset = load_from_disk(data_path).with_format('torch')
        print_rank_0(f"Loaded dataset size: {len(dataset)}")
        datasets.append(dataset)
    ds = concatenate_datasets(datasets)
    print_rank_0(f"Concatenated dataset size: {len(ds)}")
    return ds


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    return _get_dataset(args.data_path), None, None


def extra_args_provider(parser):
    parser.add_argument(
        "--rope-base",
        type=float,
        default=5000000,
    )
    parser.add_argument(
        "--sample-weights",
        nargs='*',
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default=None,
    )
    return parser


if __name__ == "__main__":
    # Temporary for transition to core datasets
    logger.info(f'开始pretrain')
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
    )

