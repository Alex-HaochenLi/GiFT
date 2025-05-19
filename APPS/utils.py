from torch.utils.data import TensorDataset, Dataset
import numpy as np
import logging
import os
import re
import random
import sys
import torch
import time
import copy
from tqdm import tqdm
from typing import *
import itertools
from _utils import *

logger = logging.getLogger(__name__)
sys.set_int_max_str_digits(60000)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def load_gen_data(args, tokenizer, split_tag, only_src=False):
    data = []
    if split_tag == 'train':
        with open(f'./data/apps+valid-{args.difficulty}.json', 'r') as f:
            data = json.load(f)

        if args.train_with_seq_gibbs:
            with open(args.selfsample_code_source, 'r') as f:
                selfsample1 = json.load(f)
            selfsample = [[] for _ in range(len(selfsample1))]
            for i in range(len(selfsample1)):
                selfsample[i].extend(selfsample1[i])

            if args.selfsample_code_source2 is not None:
                with open(args.selfsample_code_source2, 'r') as f:
                    selfsample2 = json.load(f)
                for i in range(len(selfsample2)):
                    selfsample[i].extend(selfsample2[i])

            if args.weight_file1 is not None and args.weight_file2 is not None:
                weights = [[] for _ in range(len(selfsample))]
                with open(args.weight_file1, 'r') as f:
                    weights1 = json.load(f)
                weights1 = {int(k): v for k,v in weights1.items()}
                weights1 = [weights1[key] for key in sorted(weights1.keys())]
                transformed_weights1 = []
                index = 0
                for sublist in selfsample1:
                    length = len(sublist)  # Get the length of the current sublist in list_A
                    if length > 0:
                        transformed_weights1.append(weights1[index:index + length])  # Slice list_B
                    else:
                        transformed_weights1.append([])
                    index += length  # Move the index forward by the length of the sublist

                with open(args.weight_file2, 'r') as f:
                    weights2 = json.load(f)
                weights2 = {int(k): v for k, v in weights2.items()}
                weights2 = [weights2[key] for key in sorted(weights2.keys())]
                transformed_weights2 = []
                index = 0
                for sublist in selfsample2:
                    length = len(sublist)  # Get the length of the current sublist in list_A
                    if length > 0:
                        transformed_weights2.append(weights2[index:index + length])  # Slice list_B
                    else:
                        transformed_weights2.append([])
                    index += length  # Move the index forward by the length of the sublist

                for i in range(len(selfsample)):
                    weights[i].extend(transformed_weights1[i])
                    weights[i].extend(transformed_weights2[i])
            data1 = []

            for key in range(len(selfsample)):
                ori_codes = selfsample[key]
                effective_codes = []
                for code in ori_codes:
                    docs = re.findall(f'(    \"\"\".*?\"\"\"\n)', code, re.DOTALL | re.IGNORECASE)
                    for doc in docs:
                        code = code.replace(doc, '')
                    effective_codes.append(code)

                if args.resample:
                    if args.weight_file1 is not None and args.weight_file1 is not None:
                        if len(effective_codes) > 0:
                            weight = weights[key]
                            weight = softmax(np.array(weight) * args.temperature)
                            assert len(weight) == len(effective_codes)
                            for code in random.choices(effective_codes, weights=weight, k=args.num_seq):
                                tmp = copy.deepcopy(data[key])
                                tmp['code'] = code
                                data1.append(tmp)
                    else:
                        if len(effective_codes) >= args.num_seq:
                            for code in random.sample(effective_codes, args.num_seq):
                                tmp = copy.deepcopy(data[key])
                                tmp['code'] = code
                                data1.append(tmp)
                        elif 0 < len(effective_codes) < args.num_seq:
                            for code in random.choices(effective_codes, k=args.num_seq):
                                tmp = copy.deepcopy(data[key])
                                tmp['code'] = code
                                data1.append(tmp)

                else:
                    for code in random.sample(effective_codes, min(len(effective_codes), args.num_seq)):
                        tmp = copy.deepcopy(data[key])
                        tmp['code'] = code
                        data1.append(tmp)

            save_path = f'./fixed_data/{args.difficulty}/{args.seed}'
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f'{args.output_dir}.json'), 'w') as f:
                json.dump(data1, f)
            data.extend(data1)

    elif split_tag == 'test':
        with open(f'./data/test.json', 'r') as f:
            dataset = json.load(f)
        dataset = [item for item in dataset if item['difficulty'] == args.difficulty]

        data = []
        for i, item in enumerate(dataset):
            query = re.findall(f'    \"\"\"(.*?)\"\"\"\n', item['prompt'], re.DOTALL | re.IGNORECASE)[0]
            solution = item['canonical_solution']
            js = {'query': query,
                  'test': {'fn_name': item['fn_name'], 'inputs': item['inputs'][:10], 'outputs': item['outputs'][:10]},
                  'code': solution, 'id': i, 'std_based': item['input_type']}
            data.append(js)

    features = [convert_examples_to_features(example, tokenizer, args, split_tag) for example in data]
    for idx, item in enumerate(data):
        data[idx]['source_string'] = features[idx].source_str
        data[idx]['target_string'] = features[idx].target_str
    if split_tag == 'test' or only_src:
        dataset = id_dataset([f.input_ids for f in features], list(range(len(features))), 'test')
    else:
        # remove examples that has no label due to max_length
        features = [feature for feature in features if not feature.labels[-1] == -100]
        dataset = id_dataset([f.input_ids for f in features], [f.labels for f in features], 'train')
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


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index

    return code[:min_stop_idx]