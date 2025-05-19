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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_elapse_time, load_gen_data, estimate_pass_at_k, truncate_code_at_stopwords, truncate_noisy_funcs
from configs import add_args, set_seed, set_dist
from execution import check_correctness

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate(args, eval_data, eval_examples, model, tokenizer, dataset_type):
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    data_collator = DataCollatorForTest(tokenizer)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True, collate_fn=data_collator)

    if args.local_rank == 0:
        logger.info("  " + "***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    output, tests = {}, {}
    if args.local_rank == 0:
        bar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Generating Code...")
    for idx, batch in enumerate(eval_dataloader):
        source_ids, sample_ids = batch[0].to(args.device), batch[1]
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        tmp_tests = [eval_examples[i]['test'] for i in sample_ids]
        for tmp_idx, item in enumerate(tmp_tests):
            tests[sample_ids[tmp_idx]] = item
        ori_str = [eval_examples[i]['source_string'] for i in sample_ids]
        ori_str_len = [len(item) for item in ori_str]

        with torch.no_grad():
            gen_tokens = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        do_sample=False,
                                        max_new_tokens=512,
                                        num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id,
                                        synced_gpus=True)

            tmp = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        pred_str = [tmp[i][ori_str_len[i]:] for i in range(len(tmp))]
        tmp = [truncate_code_at_stopwords(pred, ["\nclass", "\nif", "\n#", "\nprint"]) for pred in pred_str]
        tmp = [truncate_noisy_funcs(dataset_type, pred, test) for pred, test in zip(tmp, tmp_tests)]
        tmp = [ori + item for ori, item in zip(ori_str, tmp)]

        if args.local_rank == 0:
            bar.update(1)

        for tmp_idx, item in enumerate(tmp):
            output[sample_ids[tmp_idx]] = item

    if not os.path.exists('./tmp'):
        os.makedirs('./tmp', exist_ok=True)
    with open('./tmp/gen-code-{}.json'.format(args.local_rank), 'w') as f:
        json.dump(output, f)
    with open('./tmp/tests-{}.json'.format(args.local_rank), 'w') as f:
        json.dump(tests, f)

    dist.barrier()
    if args.local_rank == 0:
        bar.close()

        final_output, final_tests = {}, {}
        for i in range(dist.get_world_size()):
            # read generated codes
            file_path = os.path.join('./tmp', 'gen-code-{}.json'.format(i))
            with open(file_path, 'r') as f:
                tmp_dict = json.load(f)
                # when saving dict using json, the key will be transformed into str type, we need to recover it
                tmp_dict = {int(k): v for k, v in tmp_dict.items()}
                final_output.update(tmp_dict)
            os.remove(file_path)

            if dataset_type != 'codeinsight':
                # read tests for each sample
                file_path = os.path.join('./tmp', 'tests-{}.json'.format(i))
                with open(file_path, 'r') as f:
                    tmp_dict = json.load(f)
                    # when saving dict using json, the key will be transformed into str type, we need to recover it
                    tmp_dict = {int(k): v for k, v in tmp_dict.items()}
                    final_tests.update(tmp_dict)
                os.remove(file_path)

        output_list, test_list = [], []
        for key in sorted(final_output):
            output_list.append(final_output[key])
        if not os.path.exists(f'./gen_code/{args.output_dir}'):
            os.makedirs(f'./gen_code/{args.output_dir}')
        with open(f'./gen_code/{args.output_dir}/epoch{args.cur_epoch}.json', 'w') as f:
            json.dump(output_list, f)
        for key in sorted(final_tests):
            test_list.append(final_tests[key])
        assert len(test_list) == len(output_list)


        result = evaluate_functional_correctness(
            codes=output_list,
            tests=test_list,
            tmp_dir='./tmp',
        )
        result = {int(k): v for k, v in result.items()}
        result = [result[key][0] for key in sorted(result)]

        total, correct = [], []
        # for future in result:
            # passed = [r[1]["passed"] for r in future.result()]
        total.append(len(result))
        correct.append(sum(result))
        total = np.array(total)
        correct = np.array(correct)
        pass_at_k = {f"pass@1": estimate_pass_at_k(total, correct, 1).mean()}
        print(pass_at_k)

    dist.barrier()


def evaluate_functional_correctness(
        codes,
        tests,
        tmp_dir: str = "./",
        n_workers: int = 32,
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


def train(args, tokenizer, model):
    t0 = time.time()
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    dist.barrier()
    train_examples, train_data = load_gen_data(args, tokenizer, 'train')
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=ds_config['train_micro_batch_size_per_gpu'],
                                  num_workers=0, pin_memory=True)

    ds_config['train_batch_size'] = ds_config['train_micro_batch_size_per_gpu'] * \
                                    ds_config['gradient_accumulation_steps'] * args.world_size
    ds_config['scheduler']['params']['total_num_steps'] = len(train_data) // ds_config['train_batch_size'] \
                                                          * args.num_epoch

    model_engine, optimizer, _, _ = deepspeed.initialize(args=None, model=model,
                                                         model_parameters=model.parameters(),
                                                         config=ds_config)
    dist.barrier()

    # Start training
    train_example_num = len(train_data)
    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", ds_config['train_batch_size'])
        logger.info("  Num epoch = %d", args.num_epoch)

    for cur_epoch in range(int(args.num_epoch)):
        if args.local_rank == 0:
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        losses = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, target_ids = batch
            source_mask = source_ids.ne(tokenizer.pad_token_id)

            outputs = model_engine(input_ids=source_ids, attention_mask=source_mask,
                                   labels=target_ids)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
            losses.append(loss.item())

            if args.local_rank == 0:
                bar.update(1)

            if (step+1) % 5 and args.local_rank == 0:
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(sum(losses) / len(losses), 5)))

        if args.local_rank == 0:
            bar.close()

        args.cur_epoch = cur_epoch
        eval_examples, eval_data = load_gen_data(args, tokenizer, 'test', only_src=True)
        evaluate(args, eval_data, eval_examples, model, tokenizer)

    if args.local_rank == 0:
        if args.ckpt_path is not None:
            model_engine.module.save_pretrained(args.ckpt_path, safe_serialization=False)
            tokenizer.save_pretrained(args.ckpt_path)
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

    train(args, tokenizer, model)


if __name__ == "__main__":
    main()

