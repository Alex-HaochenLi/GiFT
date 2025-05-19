import copy
import re
import json
import random


def extract_code_head(args, example):
    code = example['code'].replace('\t', '    ').replace('\r', '')
    import_and_head = []
    for item in code.split('\n'):
        import_and_head.append(item)
        if 'def ' in item:
            break
    import_and_head = '\n'.join(import_and_head)
    source_str = f"{import_and_head}\n    \"\"\" {example['query']}\"\"\"\n"
    return source_str, import_and_head


def convert_examples_to_features(example, tokenizer, args, stage):

    if stage == 'train':
        source_str, code_head = extract_code_head(args, example)

        source_ids = tokenizer(source_str,
                               max_length=args.max_length, padding="longest",
                               truncation=True)['input_ids']

        if not args.selfsample:
            docs = re.findall(f'(    \"\"\".*?\"\"\"\n)', example['code'], re.DOTALL | re.IGNORECASE)
            for doc in docs:
                example['code'] = example['code'].replace(doc, '')
            target_str = example['code'].replace('\t', '    ').replace('\r', '').replace(code_head, '')

            example_str = source_str + target_str + tokenizer.eos_token
            input_ids = tokenizer(example_str, max_length=args.max_length,
                                   padding="longest", truncation=True)['input_ids']
            ori_length = len(input_ids)
            labels = copy.deepcopy(input_ids)
            labels[:len(source_ids)] = [-100] * len(source_ids)  # IGNORE_INDEX
            # padding from left
            input_ids = [tokenizer.pad_token_id] * (args.max_length - ori_length) + input_ids
            labels = [-100] * (args.max_length - ori_length) + labels
        else:
            input_ids = source_ids
            labels = []
            target_str = None

    elif stage == 'test':
        source_str, code_head = extract_code_head(args, example)

        input_ids = tokenizer(source_str, max_length=args.max_length, padding="longest", truncation=True)['input_ids']
        labels = []
        target_str = None


    return InputFeatures(
        example['id'],
        input_ids,
        labels,
        source_str,
        target_str
    )


def convert_examples_to_features_gibbs(ict_data, example, tokenizer, args):
    incontext_idx1 = random.randint(0, len(ict_data) - 1)
    incontext_sample1 = ict_data[incontext_idx1]

    incontext_samples = [incontext_sample1]

    source_str = construct_sum_prompt(incontext_samples, example)

    source_ids = tokenizer(source_str,
                           max_length=args.max_length, padding="longest",
                           truncation=True)['input_ids']

    input_ids = source_ids
    labels = []
    target_str = None

    return InputFeatures(
        example['id'],
        input_ids,
        labels,
        source_str,
        target_str
    )


def construct_sum_prompt(incontext_samples, example):
    incontext_str = ''
    for incontext_sample in incontext_samples:
        code = incontext_sample['code'].replace('\t', '    ').replace('\r', '')
        desc = incontext_sample['query']
        incontext_str += f"###Code:\n{code}\n###Description of the given code:\n{desc}\n\n"

    code = example['code'].replace('\t', '    ').replace('\r', '')
    source_str = f"{incontext_str}###Code:\n{code}\n###Description of the given code:\n"

    return source_str


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 input_ids,
                 labels,
                 source_str=None,
                 target_str=None
                 ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.labels = labels
        self.source_str = source_str
        self.target_str = target_str
