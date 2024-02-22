
import pickle
import json
import click
from typing import Dict, List, Set, Tuple
import os
from collections import Counter

Path = str
SRC = str
CWE = str
Label = int

def save_dict_to_pickle(dict_obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_dict_from_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def _get_stat(input_file, cwe2idx: Dict[CWE, int], cwe_stat: Counter, ignore_cwe: Set[CWE]) -> None:
    for line in input_file:
        record = json.loads(line)
        cwe = record['cwe']
        status = record['status']
        if status == 'NOT_VULNERABLE':
            cwe = status
        if cwe in ignore_cwe:
            continue
        cwe_stat[cwe] += 1
        if cwe not in cwe2idx:
            cwe2idx[cwe] = len(cwe2idx)    

def modify_jsonl(input_file, output_file, cwe2idx: Dict[CWE, int], ignore_cwe: Set[CWE]):
    for line in input_file:
        record = json.loads(line)
        cwe = record['cwe']
        status = record['status']
        if status == 'NOT_VULNERABLE':
            cwe = status
        if cwe in ignore_cwe: continue
        idx = cwe2idx.get(cwe)
        record['target'] = idx
        output_file.write(json.dumps(record) + '\n')


def get_stat(input_folder, ignore_cwe: Set[CWE]) -> Tuple[Dict[CWE, int], Dict[str, Counter]]:
    cwe2idx = dict()
    cwe_stat = dict()
    for part in ['train', 'test', 'valid']:
        stat = Counter()
        f_in_path = os.path.join(input_folder, f'{part}.jsonl')
        with open(f_in_path, 'r') as f_in:
            _get_stat(f_in, cwe2idx, stat, ignore_cwe)
        cwe_stat[part] = stat
    return cwe2idx, cwe_stat


def what_to_ignore(stat: Dict[str, Counter], ignore_cwe: List[CWE], complete: bool, min_count: int = None) -> Set[CWE]:
    if min_count is not None:
        train_cwe_ignore = {cwe for cwe, count in stat['train'].items() if count < min_count}
    
    if complete:
        train_cwe = {cwe for cwe, count in stat['train'].items()}
        test_cwe = {cwe for cwe, count in stat['test'].items()}
        valid_cwe = {cwe for cwe, count in stat['valid'].items()}
        common_cwes = train_cwe & test_cwe & valid_cwe
        all_cwes = train_cwe | test_cwe | valid_cwe
        to_ignore = all_cwes - common_cwes
    else:
        to_ignore = set()

    return set(ignore_cwe) | to_ignore | train_cwe_ignore


@click.command()
@click.option('--input_folder',  help='Path to folder with binary dataset')
@click.option('--output_folder',  help='Path to folder to save a new dataset')
@click.option('--ignore_cwe', '-i',  help='List of CWE to ignore',  multiple=True)
@click.option('--complete', is_flag=True, show_default=True, default=False, help='CWEs must be presented in all three parts of a dataset')
@click.option('--min_count',  help='Ignore CWE if its count in train part less than this number', type=int, default=0)
def main(input_folder: Path, output_folder: Path, ignore_cwe: List[CWE], complete: bool, min_count: int=None):
    "A script to convert a regular dataset for binary classification to a dataset for CWE classification"
    cwe2idx, cwe_stat = get_stat(input_folder, ignore_cwe=set())
    ignore_cwe = what_to_ignore(cwe_stat, ignore_cwe, complete, min_count)
    cwe2idx, cwe_stat = get_stat(input_folder, ignore_cwe)  # reindex dictinary and remove ignored CWE from statistics

    for part in ['train', 'test', 'valid']:
        f_in_path = os.path.join(input_folder, f'{part}.jsonl')
        f_out_path = os.path.join(output_folder, f'{part}.jsonl')
        with open(f_in_path, 'r') as f_in, open(f_out_path, 'w') as f_out:
            modify_jsonl(f_in, f_out, cwe2idx, ignore_cwe)

    if complete:
        assert len(cwe_stat['train']) == len(cwe_stat['test'])
        assert len(cwe_stat['train']) == len(cwe_stat['valid'])
    save_dict_to_pickle(cwe2idx, os.path.join(output_folder, 'cwe2idx.pickle'))
    for cwe, stat in cwe_stat.items():
        print(cwe, ':')
        print(stat)

        
    
       
       
    
if __name__ == '__main__':
    main()
