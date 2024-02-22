import json
import os
import tarfile
import tempfile

import numpy as np
import torch
from torch.utils.data import IterableDataset

def create_substring_attention_mask(input_ids, tokenizer, sep_token_id):
    #needed if we want differrent sentences not to attend each other in batch
    # for one sequence
    # Create mask with the same dimensions as input_ids
    sequencies = []  # np.zeros(len(input_ids), dtype=np.int64)
    start_new_substring = True
    pad_tokens = [sep_token_id, tokenizer.pad_token_id]
    curr_start_num = 0

    for j, token_id in enumerate(input_ids):
        if token_id in pad_tokens:  # no attention
            if not start_new_substring:
                curr_end_num = j
                sequencies.append((curr_start_num, curr_end_num))
                start_new_substring = True
        else:
            if start_new_substring:
                curr_start_num = j
                start_new_substring = False
    # Create attention mask
    mask = np.zeros((len(input_ids), len(input_ids)), dtype=np.int64)
    for (first, last) in sequencies:
        mask[first: last, first: last] = 1
    print(mask.sum(), len(sequencies))
    return mask


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        input_column_name="prompt",
        output_column_name="completion",
        concat_token_id=None,
        max_funcs_in_seq=200,
        weights=None
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.curr_idx = 0
        self.concat_token_id = concat_token_id
        self.max_funcs_in_seq = max_funcs_in_seq
        self.weights = weights

        self.seq_num = len([seq for seq in self.__iter__()])
        print("sequnce_num = ", self.seq_num)

    def reset_iter(self):
        self.indices = np.random.permutation(len(self.dataset))  # Get a random permutation of indices
        self.curr_idx = 0

    def __len__(self):
        return self.seq_num

    def __iter__(self):
        self.reset_iter()
        more_examples = True
        while more_examples:
            all_tokens, all_labels, buffer_len = [], [], 0
            if self.weights:
                batch_weights = []
            att_mask = []
            tokens_pos = []
            while True:
                if self.curr_idx < len(self.dataset):
                    curr_elem = self.dataset[self.indices[self.curr_idx]]
                    if self.curr_idx == 0:
                        print(list(curr_elem.keys()))
                    elem_len = len(curr_elem[self.input_column_name])

                    if self.curr_idx % 100 == 0:
                        print(self.curr_idx, elem_len, buffer_len, self.seq_length)
                    if elem_len >= self.seq_length:
                        self.curr_idx += 1
                        continue

                    curr_funcs_in_seq = len(all_labels)
                    if (buffer_len + elem_len >= self.seq_length) or (curr_funcs_in_seq + 1 >= self.max_funcs_in_seq):
                        # break if we have enough tokens or enough functions
                        break
                    sep_tokens = [self.concat_token_id]
                    all_tokens += curr_elem[self.input_column_name] + sep_tokens
                    att_mask.append(np.ones(elem_len + len(sep_tokens)))
                    all_labels.append(int(curr_elem[self.output_column_name]))

                    if self.weights:
                        batch_weights.append(self.weights[self.indices[self.curr_idx]])
                    buffer_len += elem_len + len(sep_tokens)
                    tokens_pos.append(buffer_len - 2)
                    self.curr_idx += 1

                else:
                    if self.infinite:
                        self.reset_iter()
                    else:
                        more_examples = False
                        break
            pad_len = (self.seq_length - len(all_tokens))
            att_mask = np.hstack(att_mask)
            att_mask = np.hstack([att_mask, np.zeros(pad_len)])
            all_tokens += [self.tokenizer.pad_token_id] * pad_len
            all_labels = np.array(all_labels)

            if all_labels.shape[0] < self.max_funcs_in_seq:
                labels_padding = np.ones(self.max_funcs_in_seq - all_labels.shape[0], dtype=int) * (-100)
                all_labels = np.hstack([all_labels, labels_padding])

                if self.weights:
                    weight_padding = labels_padding.astype(float)
                    batch_weights = np.hstack([batch_weights, weight_padding])

            outputs = {
                "input_ids": torch.LongTensor(all_tokens),
                "labels": torch.LongTensor(all_labels),
                "attention_mask": torch.LongTensor(att_mask),
                #"tokens_pos": torch.LongTensor(tokens_pos)
            }

            if self.weights:
                outputs['inputs_embeds'] = torch.FloatTensor(batch_weights)#a workaround for PEFT model not receiving another parameters

            yield outputs

class SimpleDataset(torch.utils.data.IterableDataset):
    def __init__(self, code_list, labels_list, att_list):
        self.code_list, self.labels_list, self.att_list = code_list, labels_list, att_list
        assert (len(code_list) == len(labels_list))
        assert (len(code_list) == len(att_list))

    def __len__(self):
        return len(self.code_list)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.LongTensor(self.code_list[idx]),
            "attention_mask": torch.tensor(self.att_list[idx]),
            "labels": torch.tensor(self.labels_list[idx]),
        }

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def create_datasets_for_classification(tokenizer, args, concat_token_id):
    if not args.dataset_tar_gz:
        raise NotImplementedError("specify dataset")

    tar_path = args.dataset_tar_gz
    train_file = 'train.jsonl'
    valid_file = 'valid.jsonl'
    test_file = 'test.jsonl'

    datasets = {}

    with tarfile.open(tar_path, 'r:gz') as tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar.extract(train_file, path=tmpdir)
            tar.extract(valid_file, path=tmpdir)
            tar.extract(test_file, path=tmpdir)

            for split, filename in zip(['train', 'valid', 'test'], [train_file, valid_file, test_file]):
                if not args.several_funcs_in_batch:
                    tokenized_codes = []
                    labels = []
                    masks = []

                dataelems = []

                if args.use_vul_weights:
                    weights = []
                    P3_found = False
                else:
                    weights = None
                with open(os.path.join(tmpdir, filename), 'r') as file:
                    for line in file:
                        curr_line = json.loads(line)

                        if args.delete_whitespaces:
                            curr_line['code'] = ' '.join(str(curr_line['code']).split())

                        tokenized_without_pad = tokenizer(curr_line['code'], padding=False)['input_ids']

                        if len(tokenized_without_pad) >= args.seq_length:
                            if args.ignore_large_functions:
                                continue
                            else:
                                tokenized_without_pad = tokenized_without_pad[:args.seq_length - 1]

                        curr_line['target'] = int(curr_line['target'])
                        curr_line['tokens'] = tokenized_without_pad

                        if len(curr_line['code']) == 0:
                            print("Skipping empty function")
                            continue

                        dataelems.append(curr_line)

                        if not args.several_funcs_in_batch:#if we pass functions one by one
                            tokenized_with_pad = tokenizer(curr_line['code'], padding='max_length',
                                                           max_length=args.seq_length, truncation=True)
                            # A workaround for our code of prediction - it needs pad token at the end
                            tokenized_with_pad['input_ids'][args.seq_length - 1] = tokenizer.pad_token_id
                            tokenized_with_pad['attention_mask'][args.seq_length - 1] = 0
                            tokenized_codes.append(tokenized_with_pad['input_ids'])
                            labels.append(int(curr_line['target']))
                            masks.append(tokenized_with_pad['attention_mask'])

                        if args.use_vul_weights:
                            is_P3 = (curr_line["status"] == 'NOT_VULNERABLE')
                            curr_weights = 1.0 if is_P3 else args.vul_weight
                            weights.append(curr_weights)
                            if is_P3:
                                P3_found = True

                print(f"Size of the {split} set: {len(dataelems)}.")
                if args.use_vul_weights:
                    print("P3 found: ", P3_found)
                if args.several_funcs_in_batch:
                    datasets[split] = ConstantLengthDataset(tokenizer=tokenizer, dataset=dataelems,
                                                            seq_length=args.seq_length,
                                                            input_column_name='tokens', output_column_name='target',
                                                            concat_token_id=concat_token_id,
                                                            max_funcs_in_seq=args.max_funcs_in_seq,
                                                            weights=weights
                                                            )
                else:
                    datasets[split] = SimpleDataset(code_list=tokenized_codes, labels_list=labels, att_list=masks)

    return datasets["train"], datasets["valid"], datasets['test']