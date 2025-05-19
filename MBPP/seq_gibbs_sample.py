# coding=utf-8

import os
import re
import logging
import argparse

import deepspeed
import math
import numpy as np
from tqdm import tqdm, trange
import time
import json
import random
import copy

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import truncate_code_at_stopwords, truncate_noisy_funcs
from configs import add_args, set_seed, set_dist
from _utils import *

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from execution import check_correctness

from datasets import load_dataset, concatenate_datasets

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gen_code(iter, args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data, num_replicas=args.world_size, rank=args.local_rank, shuffle=False)
    data_collator = DataCollatorForTest(tokenizer)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True, collate_fn=data_collator)

    if args.local_rank == 0:
        logger.info("  " + f"***** Generating Code Iter {iter}*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    raw_output, output, tests = {}, {}, {}
    if args.local_rank == 0:
        bar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Generating Code...")
    for idx, batch in enumerate(eval_dataloader):
        source_ids, sample_ids = batch[0].to(args.local_rank), batch[1]
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        tmp_tests = ['\n'.join(eval_examples[i]['test']) for i in sample_ids]
        tmp_tests = [element for element in tmp_tests for _ in range(args.num_seq)]

        ori_str = tokenizer.batch_decode(source_ids, skip_special_tokens=True)
        ori_str = [element for element in ori_str for _ in range(args.num_seq)]

        with torch.no_grad():
            gen_tokens = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        do_sample=True,
                                        temperature=args.temperature,
                                        max_new_tokens=512,
                                        num_return_sequences=args.num_seq,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id)

            tmp = tokenizer.batch_decode(gen_tokens[:, source_ids.size(1):], skip_special_tokens=True)

        tmp = [truncate_code_at_stopwords(pred, ["\nclass", "\nif", "\n#", "\nprint", "\nassert"]) for pred in tmp]
        tmp = [truncate_noisy_funcs('mbpp', pred, test) for pred, test in zip(tmp, tmp_tests)]
        tmp = [ori + item for ori, item in zip(ori_str, tmp)]
        assert len(tmp) == source_ids.size(0) * args.num_seq

        if args.local_rank == 0:
            bar.update(1)

        for tmp_idx, item in enumerate(tmp):
            id_of_item = eval_examples[sample_ids[tmp_idx // args.num_seq]]['id']
            if id_of_item in output:
                output[id_of_item].append(item)
            else:
                output[id_of_item] = [item]

    if not os.path.exists('./tmp'):
        os.makedirs('./tmp', exist_ok=True)
    with open('./tmp/output-{}.json'.format(args.local_rank), 'w') as f:
        json.dump(output, f)

    dist.barrier()
    if args.local_rank == 0:
        bar.close()

        save_path = './data/selfsample_code'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        output_list = gather_outputs(eval_data, raw=False)

        passed_codes = check_selfsample_code_validity(output_list)
        for i in range(len(passed_codes)):
            if len(passed_codes[i]) > 1:
                passed_codes[i] = random.sample(passed_codes[i], 1)
        assert len(passed_codes) == len(eval_examples)

        save_path = os.path.join(save_path, args.model_name_or_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, f'code_r{iter+1}.json')
        save_file(file_path, passed_codes)


def gen_sum(iter, args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    data_collator = DataCollatorForTest(tokenizer)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size * args.num_seq,
                                 num_workers=0, pin_memory=True, collate_fn=data_collator)

    if args.local_rank == 0:
        logger.info("  " + f"***** Generating Sum Iter {iter}*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size * args.num_seq)

    model.eval()
    raw_output, output, tests = {}, {}, {}
    if args.local_rank == 0:
        bar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Generating Sum...")
    for idx, batch in enumerate(eval_dataloader):
        source_ids, sample_ids = batch[0].to(args.local_rank), batch[1]
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            gen_tokens = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        do_sample=True,
                                        temperature=args.temperature,
                                        max_new_tokens=512,
                                        num_return_sequences=1,
                                        top_p=0.95,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id)

            tmp = tokenizer.batch_decode(gen_tokens[:, source_ids.size(1):], skip_special_tokens=True)

        stop_words = ["\n###Code:", "\n###"]
        tmp = [truncate_code_at_stopwords(pred, stop_words) for pred in tmp]

        if args.local_rank == 0:
            bar.update(1)

        for tmp_idx, item in enumerate(tmp):
            id_of_item = sample_ids[tmp_idx]
            if id_of_item in output:
                output[id_of_item].append(item)
                # raise Exception
            else:
                output[id_of_item] = [item]

    if not os.path.exists('./tmp'):
        os.makedirs('./tmp', exist_ok=True)
    with open('./tmp/output-{}.json'.format(args.local_rank), 'w') as f:
        json.dump(output, f)

    dist.barrier()
    if args.local_rank == 0:
        bar.close()

        save_path = './data/selfsample_sum'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        output_list = gather_outputs(eval_data, raw=False)

        save_path = os.path.join(save_path, args.model_name_or_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, f'sum_r{iter}.json')
        save_file(file_path, output_list)


def gather_outputs(eval_data, raw):
    final_output = {}
    for i in range(dist.get_world_size()):
        # read generated codes
        if not raw:
            file_path = os.path.join('./tmp', 'output-{}.json'.format(i))
        else:
            file_path = os.path.join('./tmp', 'raw-output-{}.json'.format(i))
        with open(file_path, 'r') as f:
            tmp_dict = json.load(f)
            # when saving dict using json, the key will be transformed into str type, we need to recover it
            tmp_dict = {int(k): v for k, v in tmp_dict.items()}
            for key, value in tmp_dict.items():
                if key in final_output:
                    # final_output[key].extend(value)
                    continue
                else:
                    final_output[key] = value
        os.remove(file_path)

    output_list = [[] for _ in range(len(eval_data))]
    for key in sorted(final_output):
        output_list[key].extend(final_output[key])

    return output_list


def save_file(file_path, output_list):
    with open(file_path, 'w') as f:
        json.dump(output_list, f)


def evaluate_functional_correctness(
        codes,
        tests,
        tmp_dir: str = "./",
        n_workers: int = 8,
        timeout: float = 10.0,
):
    sample = {}
    futures = []
    results = defaultdict(list)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        lang = "python"
        tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
        for i in range(len(codes)):
            sample['task_id'] = str(i)
            sample["test_code"] = codes[i] + "\n" + tests[i]
            args = (str(i), copy.deepcopy(sample), lang, timeout, tmp_dir_)
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append(result["passed"])

    return results


def check_selfsample_code_validity(codes):
    ori_dataset = load_dataset('google-research-datasets/mbpp', 'sanitized')
    dataset = concatenate_datasets([ori_dataset['train'], ori_dataset['prompt'], ori_dataset['validation']])

    tests = []
    for i in range(len(dataset)):
        tests.append(dataset[i]['test_list'])

    assert len(codes) == len(tests)

    tests_list = ['\n'.join(element) for i, element in enumerate(tests) for _ in range(len(codes[i]))]
    codes_list = [code for item in codes for code in item]
    assert len(codes_list) == len(tests_list)

    result = evaluate_functional_correctness(
        codes=codes_list,
        tests=tests_list,
        tmp_dir='./tmp',
    )
    result = {int(k): v for k, v in result.items()}
    result_list = [result[key][0] for key in sorted(result)]

    transformed_result = []
    index = 0
    for sublist in codes:
        length = len(sublist)  # Get the length of the current sublist in list_A
        transformed_result.append(result_list[index:index + length])  # Slice list_B
        index += length  # Move the index forward by the length of the sublist

    passed_code = [[] for _ in range(len(codes))]
    for i in range(len(codes)):
        for j in range(len(codes[i])):
            if transformed_result[i][j] is True:
                passed_code[i].append(codes[i][j])

    return passed_code


class DataCollatorForTest(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        source_ids = [instance[0] for instance in instances]
        source_ids = [torch.tensor(x).flip(0) for x in source_ids]
        source_ids = torch.nn.utils.rnn.pad_sequence(
            source_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        source_ids = source_ids.flip(1)  # padding from left

        return (source_ids, [instance[1] for instance in instances])


def load_gen_data(iter, args, tokenizer, split_tag='train', dataset_type='mbpp'):
    ori_dataset = load_dataset('google-research-datasets/mbpp', 'sanitized')
    dataset = concatenate_datasets([ori_dataset['train'], ori_dataset['prompt'], ori_dataset['validation']])

    data = []
    for i in range(len(dataset)):
        js = {'query': dataset[i]['prompt'], 'test': dataset[i]['test_list'], 'code': dataset[i]['code'],
              'id': i}
        data.append(js)

    if iter != 0:
        path = './data/selfsample_sum'
        path = os.path.join(path, args.model_name_or_path)
        path = os.path.join(path, f'sum_r{iter}.json')
        with open(path, 'r') as f:
            sums = json.load(f)
        assert type(sums) == list
        assert len(sums) == len(data)

        data1 = []
        for idx, item in enumerate(sums):
            tmp = copy.deepcopy(data[idx])
            tmp['query'] = item[0]
            data1.append(tmp)

        data = data1

    features = [convert_examples_to_features(example, tokenizer, args, split_tag, dataset_type) for example in data]
    for idx, item in enumerate(data):
        data[idx]['source_string'] = features[idx].source_str
        data[idx]['target_string'] = features[idx].target_str
    dataset = id_dataset([f.input_ids for f in features], list(range(len(features))), 'test')
    return data, dataset


def load_gibbs_data(iter, args, tokenizer):
    ori_dataset = load_dataset('google-research-datasets/mbpp', 'sanitized')
    dataset = concatenate_datasets([ori_dataset['train'], ori_dataset['prompt'], ori_dataset['validation']])

    data = []
    for i in range(len(dataset)):
        js = {'query': dataset[i]['prompt'], 'test': dataset[i]['test_list'], 'code': dataset[i]['code'],
              'id': i}
        data.append(js)
    # ict_data = data
    with open('./data/ict_data.json', 'r') as f:
        ict_data_raw = json.load(f)
    ict_data = []
    for id, sum in ict_data_raw:
        tmp = copy.deepcopy(data[id])
        tmp['query'] = sum
        ict_data.append(tmp)

    path = './data/selfsample_code'
    path = os.path.join(path, args.model_name_or_path)
    path = os.path.join(path, f'code_r{iter}.json')
    with open(path, 'r') as f:
        codes = json.load(f)
    assert type(codes) == list
    assert len(codes) == len(data)

    backup_code = [[] for _ in range(len(data))]
    tmp_iter = iter - 1
    while tmp_iter >= 1:
        path = './data/selfsample_code'
        path = os.path.join(path, args.model_name_or_path)
        path = os.path.join(path, f'code_r{iter}.json')
        with open(path, 'r') as f:
            tmp = json.load(f)
        for idx, item in enumerate(tmp):
            backup_code[idx].extend(item)
        tmp_iter -= 1

    data1 = []
    for i in range(len(codes)):
        assert len(codes[i]) <= 1
        if len(codes[i]) == 0:
            if len(backup_code[i]) > 0:
                code = backup_code[i][0]
            else:
                code = data[i]['code']
        else:
            code = codes[i][0]
            # if save multiple codes in selfsample codes, use this line
            # code = random.sample(codes[i], 1)[0]

        docs = re.findall(f'(    \"\"\".*?\"\"\"\n)', code, re.DOTALL | re.IGNORECASE)
        for doc in docs:
            code = code.replace(doc, '')
        tmp = copy.deepcopy(data[i])
        tmp['code'] = code
        data1.append(tmp)

    data = data1

    features = [convert_examples_to_features_gibbs(ict_data, example, tokenizer, args) for example in data]
    for idx, item in enumerate(data):
        data[idx]['source_string'] = features[idx].source_str
        data[idx]['target_string'] = features[idx].target_str
    dataset = id_dataset([f.input_ids for f in features], list(range(len(features))), 'test')

    return data, dataset


class id_dataset(Dataset):
    def __init__(self, data1, data2, phase):
        self.data1 = data1
        self.data2 = data2
        self.phase = phase

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        if self.phase == 'train':
            return (torch.tensor(item1), torch.tensor(item2))
        else:
            return (item1, item2)

    def __len__(self):
        return len(self.data1)

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

    for iter in range(args.iter_num):
        eval_examples, eval_data = load_gen_data(iter, args, tokenizer)
        gen_code(iter, args, eval_data, eval_examples, model, tokenizer)
        dist.barrier()
        eval_examples, eval_data = load_gibbs_data(iter + 1, args, tokenizer)
        gen_sum(iter + 1, args, eval_data, eval_examples, model, tokenizer)
        dist.barrier()

    if args.local_rank == 0:
        passed_codes = [[] for _ in range(len(eval_examples))]
        for i in range(1, args.iter_num+1):
            save_path = f'./data/selfsample_code'
            save_path = os.path.join(save_path, args.model_name_or_path)
            file_path = os.path.join(save_path, f'code_r{i}.json')
            with open(file_path, 'r') as f:
                tmp = json.load(f)
            for j in range(len(tmp)):
                passed_codes[j].extend(tmp[j])

        with open(os.path.join(save_path, 'passed_seq_code.json'), 'w') as f:
            json.dump(passed_codes, f)


if __name__ == "__main__":
    main()

