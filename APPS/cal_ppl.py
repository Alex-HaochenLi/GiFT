# coding=utf-8

import os
import re
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm, trange
import time
import json
import copy
from collections import defaultdict
import random
import deepspeed

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_elapse_time, estimate_pass_at_k, truncate_code_at_stopwords
from _utils import InputFeatures
from configs import add_args, set_seed, set_dist


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class id_dataset(Dataset):
    def __init__(self, sources, targets, ids):
        self.data1 = sources
        self.data2 = targets
        self.ids = ids

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.ids[index]
        return (torch.tensor(item1), torch.tensor(item2), item3)

    def __len__(self):
        return len(self.data1)

def convert_examples_to_features(example, tokenizer, args):
    source_str = example['query']

    source_ids = tokenizer(source_str,
                           max_length=args.max_length, padding="longest",
                           truncation=True)['input_ids']

    target_str = example['code'].replace('\t', '    ').replace('\r', '')

    example_str = source_str + target_str + tokenizer.eos_token
    input_ids = tokenizer(example_str, max_length=args.max_length,
                           padding="longest", truncation=True)['input_ids']
    ori_length = len(input_ids)
    labels = copy.deepcopy(input_ids)
    labels[:len(source_ids)] = [-100] * len(source_ids)  # IGNORE_INDEX
    # padding from left
    input_ids = [tokenizer.pad_token_id] * (args.max_length - ori_length) + input_ids
    labels = [-100] * (args.max_length - ori_length) + labels


    return InputFeatures(
        example['id'],
        input_ids,
        labels,
        source_str,
        target_str
    )


def load_gen_data(args, tokenizer):
    data = []
    i = 0
    with open(args.selfsample_code_source, 'r') as f:
        selfsample = json.load(f)

    for key in range(len(selfsample)):
        codes = selfsample[key]
        for code in codes:
            matches = re.findall(r"(.*?)(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:\s*\"\"\"[\s\S]*?\"\"\")", code, re.DOTALL | re.IGNORECASE)
            query = ''.join(matches[0])
            js = {'query': query, 'code': code.replace(query, ''), 'id': i}
            i += 1
            data.append(js)

    features = [convert_examples_to_features(example, tokenizer, args) for example in data]
    for idx, item in enumerate(data):
        data[idx]['source_string'] = features[idx].source_str
        data[idx]['target_string'] = features[idx].target_str

    dataset = id_dataset([f.input_ids for f in features], [f.labels for f in features], [f.example_id for f in features])
    return data, dataset


def train(args, tokenizer, model):
    t0 = time.time()

    dist.barrier()
    train_examples, train_data = load_gen_data(args, tokenizer)
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.eval_batch_size,
                                  num_workers=0, pin_memory=True)

    # Start training
    train_example_num = len(train_data)
    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.eval_batch_size)

    if args.local_rank == 0:
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
    model.eval()
    output = {}
    for step, batch in enumerate(train_dataloader):
        source_ids, target_ids = batch[0].to(args.device), batch[1].to(args.device)
        sample_ids = batch[2].item()
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
            loss = outputs.loss

        perplexity = torch.exp(loss).item()

        if args.local_rank == 0:
            bar.update(1)

        output[sample_ids] = perplexity

    if args.local_rank == 0:
        bar.close()

    if not os.path.exists('./tmp'):
        os.makedirs('./tmp', exist_ok=True)
    with open('./tmp/ppl-{}.json'.format(args.local_rank), 'w') as f:
        json.dump(output, f)

    dist.barrier()
    if args.local_rank == 0:
        final_output = {}
        for i in range(dist.get_world_size()):
            # read generated codes
            file_path = os.path.join('./tmp', 'ppl-{}.json'.format(i))
            with open(file_path, 'r') as f:
                tmp_dict = json.load(f)
                # when saving dict using json, the key will be transformed into str type, we need to recover it
                tmp_dict = {int(k): v for k, v in tmp_dict.items()}
                final_output.update(tmp_dict)
            os.remove(file_path)

        with open(os.path.join(os.path.dirname(args.selfsample_code_source), 'ppl-'+os.path.basename(args.selfsample_code_source)), 'w') as f:
            json.dump(final_output, f)

    if args.local_rank == 0:
        logger.info("Finish training and take %s", get_elapse_time(t0))


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)

    set_dist(args)
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left',
                                              use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = deepspeed.init_inference(model, dtype=torch.bfloat16)

    train(args, tokenizer, model)


if __name__ == "__main__":
    main()

