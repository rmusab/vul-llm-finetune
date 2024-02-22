import argparse
import json
import logging
import sys
import tarfile
import tempfile
import shutil
import math

import numpy as np
import scipy.special
import torch
from accelerate import Accelerator
from torch.utils.data import IterableDataset
from tqdm import tqdm

import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          EvalPrediction, GPTBigCodeConfig,
                          GPTBigCodeForSequenceClassification, Trainer,
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict, AdaLoraConfig

sys.path.append("/home/ma-user/modelarts/inputs/code_1")
from finetune.dataset import create_datasets_for_classification
from finetune.gpt_big_code_classification_several_funcs import GPTBigCodeClassificationSeveralFunc, GPTBigCodeConfigClassificationSeveralFunc
from utils.calc_quality import quality_short_report_val, quality_full_report_val
from debug_funcs import _build_debug_param_to_name_mapping_our_debug, debug_params


transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class SaveBestModelCallback(TrainerCallback):
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        # Get the current AUC from the metrics
        current_auc = metrics['eval_roc_auc']

        # Compare it to the best AUC so far (if any)
        if state.best_metric is None or current_auc > state.best_metric:
            state.best_metric = current_auc
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{round(state.epoch)}")

            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            torch.save({}, pytorch_model_path)

            print(f"New best model found! Saving model with AUC {state.best_metric} to {checkpoint_folder}")

            if hasattr(state, 'best_model_checkpoint') and state.best_model_checkpoint != None and os.path.exists(state.best_model_checkpoint):
                try:
                    shutil.rmtree(state.best_model_checkpoint)
                except Exception as e:
                    print("Error asd:", e)

            # Save the path to the best model
            state.best_model_checkpoint = checkpoint_folder

class LoadBestModelCallback(TrainerCallback):
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.best_model_checkpoint is not None:
            print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
            best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
            adapters_weights = torch.load(best_model_path)
            model = kwargs["model"]
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print("No best model checkpoint available.")
        return control

def get_args():
    parser = argparse.ArgumentParser()

    # Paths and data related arguments
    parser.add_argument("--model_path", type=str, default="/home/ma-user/modelarts/inputs/model_2/")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/CodeAlpaca_20K")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--input_column_name", type=str, default="code")
    parser.add_argument("--output_column_name", type=str, default="target")
    parser.add_argument("--dataset_tar_gz", default=None, type=str)
    parser.add_argument("--data", default=None, type=str)
    parser.add_argument("--code", default=None, type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--results", default=None, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--base_model", default='starcoder', type=str)

    # Model parameters
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eos_token_id", type=int, default=49152)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Learning parameters
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", default=10, type=int)

    # Runtime and logging parameters
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)  # V100 don't support
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)

    # Additional settings
    parser.add_argument("--delete_whitespaces", action="store_true", default=False)
    parser.add_argument("--ignore_large_functions", action="store_true", default=False)
    parser.add_argument("--several_funcs_in_batch", action="store_true", default=False)
    parser.add_argument("--debug_on_small_model", action="store_true", default=False)
    parser.add_argument("--max_funcs_in_seq", default=200, type=int)
    parser.add_argument("--use_adalora", action="store_true", default=False)
    parser.add_argument("--use_focal_loss", action="store_true", default=False)
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0)
    parser.add_argument("--loss_reduction", type=str, default='sum')
    parser.add_argument("--run_test", action="store_true", default=False)
    parser.add_argument("--run_test_peft", action="store_true", default=False)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)

    parser.add_argument("--use_vul_weights", action="store_true", default=False)
    parser.add_argument("--vul_weight", type=float, default=1.0)
    parser.add_argument("--top_mlp_layers_num", default=0, type=int)

    parser.add_argument("--lora_bias", type=str, default='none')

    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class EvalQuality():
    def __init__(self, metric=None):
        self.metric = metric

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        labels_vect = []
        logits_vect = []

        if len(labels.shape) == 1:
            for i in range(labels.shape[0]):
                if labels[i] >= 0:
                    labels_vect.append(labels[i])
                    logits_vect.append(logits[i, :].ravel())
        elif len(labels.shape) == 2:
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i, j] >= 0:
                        labels_vect.append(labels[i, j])
                        logits_vect.append(logits[i, j, :].ravel())
        else:
            raise(NotImplementedError("lavels shape doesn't match"))

        labels_vect = np.array(labels_vect)
        logits_vect = np.vstack(logits_vect)
        probs_pos_class = scipy.special.softmax(logits_vect, axis=-1)[:, 1]
        full_report = quality_full_report_val(probs_pos_class, labels_vect)
        print(full_report)
        print(f"dataset size on evaluation: {len(labels_vect)}")
        report = full_report[2]
        if self.metric:
            return report[self.metric]
        else:
            return report


def create_small_gptbigcode_config(tokenizer):
    config = GPTBigCodeConfig(
        vocab_size=len(tokenizer),  # Размер словаря
        n_positions=2048,  # Максимальное количество позиций
        n_embd=10,  # Размерность вложений и скрытых состояний
        n_layer=2,  # Количество скрытых слоев в кодировщике Transformer
        n_head=2,  # Количество голов внимания для каждого слоя внимания в кодировщике Transformer
        n_inner=None,  # Размерность внутренних слоев feed-forward. None установит его в 4 раза больше n_embd
        activation_function="gelu_pytorch_tanh",  # Функция активации
        resid_pdrop=0.1,  # Вероятность dropout для всех полносвязных слоев в вложениях, кодировщике и пулере
        embd_pdrop=0.1,  # Вероятность dropout для вложений
        attn_pdrop=0.1,  # Вероятность dropout для внимания
        layer_norm_epsilon=1e-5,  # Эпсилон для слоев нормализации
        initializer_range=0.02,
        # Стандартное отклонение truncated_normal_initializer для инициализации всех матриц весов
        scale_attn_weights=True,  # Масштабировать веса внимания, разделив на sqrt(hidden_size)
        use_cache=True,
        # Должна ли модель возвращать последние ключи/значения внимания (не используется всеми моделями)
        attention_softmax_in_fp32=True,  # Вызывать ли слияние softmax в float32
        scale_attention_softmax_in_fp32=True,  # Масштабировать ли softmax внимания в float32
        attention_type=True,  # Использовать ли Multi-Query Attention (True) или Multi-Head Attention (False)
        num_labels=2
    )
    return config


def get_sep_token_id(tokenizer, args):
    if tokenizer.sep_token_id is not None:
        return tokenizer.sep_token_id
    elif tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    else:
        return args.eos_token_id

def prepare_model_and_data(args):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    # model = AutoModelForCausalLM.from_pretrained(
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sep_token_id = get_sep_token_id(tokenizer, args)

    """add special tokens"""
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    ModelClass = GPTBigCodeClassificationSeveralFunc \
        if  args.several_funcs_in_batch\
        else GPTBigCodeForSequenceClassification

    if args.debug_on_small_model:
        config = create_small_gptbigcode_config(tokenizer)
    else:
        config_class = GPTBigCodeConfigClassificationSeveralFunc if args.several_funcs_in_batch else GPTBigCodeConfig
        config, model_kwargs = config_class.from_pretrained(args.model_path,
            use_cache=not args.no_gradient_checkpointing,
            load_in_8bit=True,
            device_map={"": Accelerator().process_index},
            num_labels=2, return_unused_kwargs=True)

    if args.several_funcs_in_batch:
        args.sep_token_id = sep_token_id
        config.set_special_params(args)

    config.pad_token_id = tokenizer.eos_token_id
    config.pad_token = tokenizer.eos_token

    if args.debug_on_small_model:
        model = ModelClass(config)
    else:
        model = ModelClass.from_pretrained(
            args.model_path,
            config=config,
            **model_kwargs
        )
    train_data, val_data, test_data = create_datasets_for_classification(tokenizer, args, sep_token_id)

    return {"model": model, "tokenizer":tokenizer, "data": (train_data, val_data, test_data)}

def prepare_peft_model(model, args):
    for name, module in model.named_modules():
        print(f"{name} : {type(module).__name__}", flush=True)
    model = prepare_model_for_int8_training(model)

    for name, module in model.named_modules():
        print(f"{name} : {type(module).__name__}", flush=True)

    if str(args.base_model).lower() == "codegen2":
        target_modules = ["qkv_proj", "out_proj", "fc_in", "fc_out"]
        modules_to_save = ['lm_head']
    elif str(args.base_model).lower() == "starcoder":
        target_modules = ["c_proj", "c_attn", "c_fc", "linear_layer"]  # , 'wpe', 'wpe'
        for i in range(args.top_mlp_layers_num):
            target_modules.append(f"linear_layer_{i}")
        modules_to_save = None  # ["score"]#, 'wpe', 'wte']#, 'ln_1', 'ln_2', 'ln_f'
    else:
        raise NotImplementedError(args.base_model)

    config_class = AdaLoraConfig if args.use_adalora else LoraConfig

    lora_config = config_class(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,  # Starcoder
        modules_to_save=modules_to_save,
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model

def prepare_trainer(model, train_data, val_data, args):
    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=2,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-finetuned",
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_train_epochs,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
                      compute_metrics=EvalQuality(),
                      callbacks=[LoadBestModelCallback, SaveBestModelCallback])
    # callbacks = [SavePeftModelCallback, LoadBestPeftModelCallback])
    return trainer

def run_training(args):
    model_and_data = prepare_model_and_data(args)
    model = model_and_data['model']
    tokenizer = model_and_data['tokenizer']
    train_data, val_data, test_data = model_and_data['data']
    model = prepare_peft_model(model, args)
    trainer = prepare_trainer(model, train_data, val_data, args)

    print("Training...")
    # debug_params(trainer)
    trainer.train()
    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    # results = trainer.predict(test_data)
    # print(f"Test results...")
    # print(results.metrics)


def run_test_peft(args):
    model_and_data = prepare_model_and_data(args)
    model = model_and_data['model']
    tokenizer = model_and_data['tokenizer']
    train_data, val_data, test_data = model_and_data['data']
    model = prepare_peft_model(model, args)
    trainer = prepare_trainer(model, train_data, val_data, args)
    test_checkpoint_path = trainer.state.best_model_checkpoint
    if args.model_checkpoint_path:
        test_checkpoint_path = os.path.join(args.checkpoint_dir, args.model_checkpoint_path)

    if test_checkpoint_path:
        print(f"Loading: [{test_checkpoint_path}]...")
        best_model_path = os.path.join(test_checkpoint_path, "adapter_model.bin")
        print(os.path.exists(test_checkpoint_path))
        print(os.path.exists(best_model_path))
        adapters_weights = torch.load(best_model_path)
        set_peft_model_state_dict(model, adapters_weights)

    results = trainer.predict(test_data)
    print(f"Test results...")
    print(results.metrics)


def run_test(args):
    model_and_data = prepare_model_and_data(args)
    model = model_and_data['model']
    tokenizer = model_and_data['tokenizer']
    train_data, val_data, test_data = model_and_data['data']
    trainer = prepare_trainer(model, train_data, val_data, args)
    results = trainer.predict(test_data)
    print(f"Test results...")
    print(results.metrics)

def main(args):
    if not args.run_test and not args.run_test_peft:
        run_training(args)
    elif args.run_test:
        run_test(args)
    elif args.run_test_peft:
        run_test_peft(args)
    else:
        raise NotImplementedError("")


if __name__ == "__main__":
    import torch
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    args = get_args()
    set_seed(args.seed)
    main(args)
