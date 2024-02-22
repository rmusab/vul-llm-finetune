from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default="/")

    parser.add_argument("--data", default=None, type=str)
    parser.add_argument("--code", default=None, type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--peft_path", default=None, type=str)

    return parser.parse_args()

def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16 
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = f"{args.base_model_name_or_path}-merged"

    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(save_path, use_temp_dir=False, private=True)
        tokenizer.push_to_hub(save_path, use_temp_dir=False, private=True)
    else:
        print(f"Saving to {save_path} ...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__ == "__main__" :
    main()
