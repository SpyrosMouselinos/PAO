import numpy as np
from typing import List, Optional
import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import (LengthGroupedSampler, RandomSampler,
                                  has_length)
from transformers.trainer_pt_utils import logger
import json
import os
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled
import socket
import subprocess
from datetime import timedelta
# import deepspeed
import torch.multiprocessing as mp
from torch import distributed as dist

timeout = timedelta(minutes=60)
IGNORE_INDEX = -100


def concat_pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


def replace_llama_rmsnorm_with_fused_rmsnorm():
    try:
        from functools import partial

        from apex.normalization import FusedRMSNorm
        LlamaRMSNorm = partial(FusedRMSNorm, eps=1e-6)  # noqa
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        print('Discovered apex.normalization.FusedRMSNorm - will use it instead of LlamaRMSNorm')
    except ImportError:
        # using the normal LlamaRMSNorm
        pass
    except Exception:
        print('discovered apex but it failed to load, falling back to LlamaRMSNorm')
        pass


# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks


# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            dataset: Optional[Dataset] = None,
            lengths: Optional[List[int]] = None,
            model_input_name: Optional[str] = None,
            generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            # self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)


def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
    print('Replace train sampler!!')


def get_num_layer_for_vit_and_qllama(var_name, vit_num_max_layer, llama_num_max_layer):
    if var_name.startswith('internvl.'):
        var_name = var_name[len('internvl.'):]
    if var_name in ('query_tokens', 'logit_scale',):
        return 0
    if var_name.startswith('clip_projector.'):
        return vit_num_max_layer
    if var_name.startswith('clip_projector2.') or var_name.startswith('itm_head.') or \
            var_name == 'text_projection':
        return llama_num_max_layer
    if var_name.startswith('vision_model.'):
        if 'embeddings.' in var_name:
            return 0
        if 'layers.' in var_name:
            var_name = var_name.split('layers.')[-1]
            layer_id = int(var_name.split('.')[0])
            return layer_id + 1
    if var_name.startswith('qllama.'):
        if 'embed_tokens' in var_name:
            return 0
        if 'layers.' in var_name:
            var_name = var_name.split('layers.')[-1]
            layer_id = int(var_name.split('.')[0])
            return layer_id + 1
        else:
            return llama_num_max_layer
    return 0


def param_classification(name):
    if name.startswith('internvl.'):
        name = name[len('internvl.'):]
    if name in ['query_tokens', 'text_projection', 'logit_scale']:
        return 'qllama'
    elif name.startswith('vision_model.'):
        return 'vit'
    elif name.startswith('qllama.'):
        return 'qllama'
    elif name.startswith('clip_projector.'):
        return 'vit'
    elif name.startswith('clip_projector2.'):
        return 'qllama'
    elif name.startswith('itm_head.'):
        return 'qllama'
    else:
        return 'other'


def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    parameter_groups = {}
    try:  # for stage2 model
        vit_num_layers = opt_model.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.config.qllama_config.num_hidden_layers + 2
    except:  # for stage3 model
        vit_num_layers = opt_model.internvl.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.internvl.config.qllama_config.num_hidden_layers + 2
    print('vit_num_layers:', vit_num_layers)
    print('qllama_num_layers:', qllama_num_layers)

    vit_layer_decay_rate = float(os.getenv('VIT_LAYER_DECAY_RATE', 1.0))
    qllama_layer_decay_rate = float(os.getenv('QLLAMA_LAYER_DECAY_RATE', 1.0))
    qllama_lr_scale = float(os.getenv('QLLAMA_LR_SCALE', 1.0))
    print('vit_layer_decay_rate:', vit_layer_decay_rate)
    print('qllama_layer_decay_rate:', qllama_layer_decay_rate)
    print('qllama_lr_scale:', qllama_lr_scale)

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = self.args.weight_decay

        cls = param_classification(name)
        layer_id = get_num_layer_for_vit_and_qllama(name, vit_num_layers, qllama_num_layers)
        group_name = '%s_layer_%d_%s' % (cls, layer_id, group_name)
        if group_name not in parameter_groups:
            if cls == 'vit':
                scale = vit_layer_decay_rate ** (vit_num_layers - layer_id - 1)
            elif cls == 'qllama':
                scale = qllama_layer_decay_rate ** (qllama_num_layers - layer_id - 1)
                scale = scale * qllama_lr_scale
            else:
                scale = 1.0
            scale = min(1.0, scale)
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * self.args.learning_rate,
            }
        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

        rank = torch.distributed.get_rank()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

    optimizer_grouped_parameters = list(parameter_groups.values())
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == 'Adam8bit':
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f'skipped {module}: {skipped / 2 ** 20}M params')
                manager.register_module_override(module, 'weight', {'optim_bits': 32})
                logger.debug(f'bitsandbytes: will optimize {module} in fp32')
        logger.info(f'skipped: {skipped / 2 ** 20}M params')

    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        self.optimizer = smp.DistributedOptimizer(self.optimizer)

    return self.optimizer


def replace_create_optimizer():
    print('Replace original create_optimizer with custom create_optimizer')
    transformers.Trainer.create_optimizer = create_optimizer


def _find_free_port():
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port):
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        raise NotImplementedError(f'Invalid launcher type: {launcher}')
        #_init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        raise NotImplementedError(f'Invalid launcher type: {launcher}')
        #_init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_mpi(backend, **kwargs):
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)

### UNUSED SECTION ###


# def _init_dist_pytorch(backend, **kwargs):
#     # TODO: use local_rank instead of rank % num_gpus
#     rank = int(os.environ['RANK'])
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(rank % num_gpus)
#     # dist.init_process_group(backend=backend, **kwargs)
#     deepspeed.init_distributed(dist_backend=backend)
#
#
# def _init_dist_slurm(backend, port=None):
#     """Initialize slurm distributed training environment.
#
#     If argument ``port`` is not specified, then the master port will be system
#     environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
#     environment variable, then a default port ``29500`` will be used.
#
#     Args:
#         backend (str): Backend of torch.distributed.
#         port (int, optional): Master port. Defaults to None.
#     """
#     proc_id = int(os.environ['SLURM_PROCID'])
#     ntasks = int(os.environ['SLURM_NTASKS'])
#     node_list = os.environ['SLURM_NODELIST']
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(proc_id % num_gpus)
#     addr = subprocess.getoutput(
#         f'scontrol show hostname {node_list} | head -n1')
#     # specify master port
#     if port is not None:
#         os.environ['MASTER_PORT'] = str(port)
#     elif 'MASTER_PORT' in os.environ:
#         pass  # use MASTER_PORT in the environment variable
#     else:
#         # if torch.distributed default port(29500) is available
#         # then use it, else find a free port
#         if _is_free_port(29500):
#             os.environ['MASTER_PORT'] = '29500'
#         else:
#             os.environ['MASTER_PORT'] = str(_find_free_port())
#     # use MASTER_ADDR in the environment variable if it already exists
#     if 'MASTER_ADDR' not in os.environ:
#         os.environ['MASTER_ADDR'] = addr
#     os.environ['WORLD_SIZE'] = str(ntasks)
#     os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
#     os.environ['RANK'] = str(proc_id)
#     # dist.init_process_group(backend=backend, timeout=timeout)
#     deepspeed.init_distributed(dist_backend=backend)
