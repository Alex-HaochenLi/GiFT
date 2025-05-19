import os
import random
import datetime
import torch
import logging
import multiprocessing
import numpy as np
import deepspeed
from torch import distributed as dist

logger = logging.getLogger(__name__)


def add_args(parser):

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--max_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_epoch", type=int, required=False,
                        help="")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank.")
    parser.add_argument('--seed', type=int, default=1234,
                        help="Random seed for initialization.")
    parser.add_argument('--deepspeed_config', type=str, required=True,
                        help="Config path for DeepSpeed.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Decide the saving file of generated code either in self-sampling or in fine-tuning.")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path to load/save checkpoint.")

    parser.add_argument("--selfsample", action='store_true',
                        help="Choose whether to sample from LLM's output.")
    parser.add_argument("--train_with_seq_gibbs", action='store_true',
                        help="Choose whether to fine-tune LLMs with self-generated codes.")

    parser.add_argument('--selfsample_code_source', type=str, default=None,
                        help="Decide which selfsample data source to load.")
    parser.add_argument('--selfsample_code_source2', type=str, default=None,
                        help="Decide which selfsample data source to load.")
    parser.add_argument('--num_seq', type=int, default=1,
                        help="In self-generation: Number of sequences to be generated for one prompt."
                             "In fine-tuning: Select K codes paired with each description")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="In self-generation: Temperature for LLM generation."
                             "In fine-tuning: Temperature for perplexity weights.")
    parser.add_argument('--iter_num', type=int, default=50,
                        help="Iteration numbers for self-sampling.")

    parser.add_argument("--resample", action='store_true',
                        help="Choose whether to resample self-generated codes if candidate numbers are less than K.")
    parser.add_argument('--difficulty', type=str, required=True,
                        help="introductory/interview")
    parser.add_argument('--weight_file1', type=str, default=None,
                        help="Decide which perplexity weight source to load.")
    parser.add_argument('--weight_file2', type=str, default=None,
                        help="Decide which perplexity weight source to load.")

    args = parser.parse_args()

    args.lang = 'python'
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.world_size = 1
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=9600))
        args.world_size = dist.get_world_size()
        args.n_gpu = args.world_size
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
