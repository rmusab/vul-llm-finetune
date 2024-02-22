from datetime import datetime
import json
import os
import logging as log
from pathlib import Path
import tarfile
from typing import List, Tuple
import warnings

import click
import scipy
from guidance import Program
from guidance.llms import Transformers
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm
from transformers import logging as hf_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def load_samples_from_tar_gz(file_path: str, file_in_archive: str = "test.jsonl") -> Tuple[List[str], List[str]]:
    """"
    Json has several fields:
    * code: string value like "public static String URLDecode(String str) {\n        return..."
    * idx: integer value like 78578
    * cwe: string like "CWE-79"
    * target: integer value where 1 means that it has vulnerability, 0 - no vulnerability
    """
    functions, idxs, cwes, is_vulnerable = [], [], [], []
    with tarfile.open(file_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(file_in_archive):
                f = tar.extractfile(member)
                for line in f:
                    data = json.loads(line)
                    functions.append(data['code'])
                    idxs.append(data["idx"])
                    cwes.append(data["cwe"])
                    is_vulnerable.append('YES' if data['target'] == 1 else 'NO')
                break

    return functions, idxs, cwes, is_vulnerable


def double_print(text):
    log.info(text)
    print(text, flush=True)


def is_short_prompt(prompt, tokenizer, llm):
    yes_prompt = prompt + "YES"
    no_prompt = prompt + "NO"
    max_position_embeddings = llm.config.max_position_embeddings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if (len(tokenizer.encode(yes_prompt)) >= max_position_embeddings or
                len(tokenizer.encode(no_prompt)) >= max_position_embeddings):
            return False
    return True


def quality_report(ground_truth, predictions, yes_proba, pos_label="YES"):
    double_print(f"Metrics after {len(predictions)} samples")
    double_print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    double_print(f"Precision: {precision_score(ground_truth, predictions, pos_label=pos_label, zero_division=0)}")
    double_print(f"Recall: {recall_score(ground_truth, predictions, pos_label=pos_label, zero_division=0)}")
    double_print(f"F1: {f1_score(ground_truth, predictions, pos_label=pos_label, zero_division=0)}")
    try:
        roc_auc = roc_auc_score([1 if v == pos_label else 0 for v in ground_truth], yes_proba)
        double_print(f"roc_auc_score: {roc_auc}")
    except ValueError:
        double_print("No positive labels - no roc_auc_score")
        pass


def calc_yes_probability(yes_logproba, no_logproba):
    return np.exp(yes_logproba) / (np.exp(yes_logproba) + np.exp(no_logproba))


def preprocess_function(function):
    return function.replace('{{', '{ {').replace('}}', '} }')


def prepare_prompt_star_format(function: str) -> str:
    prompt = f"Question: {' '.join(str(function).split())}\n\nAnswer: "
    return prompt


def are_vulnerable_funcs(functions: List[str], idxs: List[int], model_loc: str, ground_truth: List[str],
                         options: List[str]):
    """
    Predicts whether each function from a list is vulnerable or not using a pre-trained tokenizer and model.
    :param functions: List of code snippets/functions
    :param idxs: List of function ids
    :param model_loc: location of the model - URL or local directory
    :param ground_truth: List of ground truth labels
    :return: List of predictions corresponding to each input function ("YES" if vulnerable, "NO" otherwise)
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_loc, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    processed_idxs = []
    predictions = []
    yes_proba = []
    gt = []
    n_skipped_functions = 0

    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens=1,
        max_new_tokens=3,
    )

    for i, (function, label, idx) in enumerate(tqdm(zip(functions, ground_truth, idxs), total=len(functions),
                                               desc="Predicting", dynamic_ncols=True)):
        # prompt = func_nearest_neighbor.prepare_prompt(function=function, llm=llm)
        prompt = prepare_prompt_star_format(function=function)
        if is_short_prompt(prompt=prompt, tokenizer=tokenizer, llm=model):
            # send only promts that fit into LLM context size
            try:
                batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
                # update generated ids, model inputs, and length for next step
                # """
                # token_tensor = torch.tensor([[225]]).to("cuda")
                #
                # input_ids = torch.cat([batch['input_ids'], token_tensor], dim=-1)
                # model_inputs = model.prepare_inputs_for_generation(input_ids)
                # """
                # """
                # generated_ids = model.generate(**batch, generation_config=generation_config)
                #
                # yes_id = tokenizer.convert_tokens_to_ids(options[0])
                # no_id = tokenizer.convert_tokens_to_ids(options[1])
                # # yes_id = tokenizer.convert_tokens_to_ids("YES")
                # # no_id = tokenizer.convert_tokens_to_ids("NO")
                # generated_text = []
                # for idx in generated_ids[0]:
                #     generated_token = tokenizer.decode(idx, skip_special_tokens=False)
                #     generated_text.append((idx, generated_token))
                # double_print("======================")
                # double_print(generated_text)
                # double_print("======================")
                # """
                logits = model(**batch).logits.cpu().detach().numpy()
                # """
                # answer_indices = np.argmax(logits, axis=-1, keepdims=False)
                #
                # for i in range(answer_indices.shape[0]):
                #     double_print("batch " + str(i))
                #     for j in range(answer_indices.shape[1]):
                #         double_print("generated token = '" + str(tokenizer.decode([answer_indices[i, j]])) + "'")
                # """
                second_token_logits = logits[0, -2, :]
                yes_logit = second_token_logits[24065]
                no_logit = second_token_logits[4435]
                # double_print(f"yes_logit: {yes_logit}")
                # double_print(f"no_logit: {no_logit}")
                curr_probas = scipy.special.softmax(np.array([yes_logit, no_logit]))
                yes_proba.append(curr_probas[0])
                # double_print(f"\ncalc_yes_probability: {curr_probas}\n")
                predictions.append(options[0] if curr_probas[0] > 0.5 else options[1])
                pref_label = ""
                if options[0][0] != "Y":
                    pref_label = options[0][0]
                gt.append(pref_label + label)
                processed_idxs.append(idx)
                """
                else:
                    double_print("-" * 100)
                    double_print(generated_text)
                    double_print("-" * 100)
                    n_skipped_functions += 1
                """
            except Exception as e:
                double_print("-" * 100)
                double_print(str(e))
                double_print("-" * 100)
                n_skipped_functions += 1
        else:
            n_skipped_functions += 1
        if (i + 1) % 100 == 0:
            # print(f"predictions: {predictions}\n\ngt: {gt}")
            quality_report(ground_truth=gt, predictions=predictions, yes_proba=yes_proba, pos_label=options[0])
            double_print(f"Number of skipped functions: {n_skipped_functions}")
    double_print(f"Number of skipped functions: {n_skipped_functions}")
    return predictions, yes_proba, gt, processed_idxs


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Path to tar.gz file containing test data')
@click.option('-m', '--model', 'model_name', default="TheBloke/Wizard-Vicuna-13B-Uncensored-HF", type=str,
              help='Name of the pretrained model to be used')
@click.option('-f', '--file-in-archive', 'file_in_archive', default="test.jsonl", type=str,
              help='File to use in archive')
@click.option('-o', '--output', 'output', required=True, type=click.Path(exists=True),
              help='Directory to store results. It will create file with name "yyyy-mm-dd-HH-MM-SS"')
def main(input_file: str, model_name: str, output: str, file_in_archive: str = "test.jsonl"):
    output = Path(output)
    test_functions, test_idxs, test_cwes, test_ground_truth = load_samples_from_tar_gz(
        file_path=input_file,
        file_in_archive="test.jsonl"
    )

    options = ["ĠYES", "ĠNO"]
    options = [" YES", " NO"]
    # options = ["YES", "NO"]


    current_time_filename = prepare_output_filename(file_in_archive, input_file)
    predictions, yes_proba, gt, processed_idxs = are_vulnerable_funcs(
        functions=test_functions, model_loc=model_name, ground_truth=test_ground_truth, idxs=test_idxs,
        options=options
    )
    quality_report(ground_truth=gt, predictions=predictions, yes_proba=yes_proba, pos_label=options[0])

    with open(output / current_time_filename, "w") as f:
        for true_proba, idx in zip(yes_proba, processed_idxs):
            f.write(str(idx) + "\t" + str(true_proba) + "\n")


def prepare_output_filename(file_in_archive, input_file):
    arch = os.path.splitext(Path(input_file).stem)[0]  # name of archive without extension
    f_in_arch = os.path.splitext(Path(file_in_archive).stem)[0]  # name of file in archive without extension
    current_time_filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "__" + arch + "__" + f_in_arch + ".txt"
    return current_time_filename


if __name__ == '__main__':
    hf_logging.set_verbosity(hf_logging.ERROR)
    hf_logging.get_logger().setLevel(hf_logging.ERROR)
    log.basicConfig(level="INFO")
    main()
