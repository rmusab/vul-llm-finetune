# Finetuning Large Language Models for Vulnerability Detection

## Parameters

This folder contains a program providing the code for finetuning of LLM models StarCoder and WizardCoder for the task of vulnerability detection. The program accepts several command-line arguments that can be used to customize its behavior. Here is a brief explanation of each argument:

- `model_path`: a string specifying the path to the model directory.
- `dataset_name`: a string specifying the name of the dataset.
- `subset`: a string specifying the name of the subset of the dataset to use.
- `split`: a string specifying the name of the split of the dataset to use.
- `streaming`: a boolean flag indicating whether to use streaming or not.
- `shuffle_buffer`: an integer specifying the size of the shuffle buffer.
- `input_column_name`: a string specifying the name of the input column in the dataset.
- `output_column_name`: a string specifying the name of the output column in the dataset.
- `dataset_tar_gz`: a string specifying the path to the dataset tar.gz file.
- `data`: a string specifying the path to the data directory.
- `code`: a string specifying the path to the code directory.
- `model`: a string specifying the path to the model directory.
- `results`: a string specifying the path to the results directory.
- `checkpoint_dir`: a string specifying the path to the checkpoint directory.
- `base_model`: a string specifying the name of the base model to use.
- `seq_length`: an integer specifying the sequence length.
- `batch_size`: an integer specifying the batch size.
- `gradient_accumulation_steps`: an integer specifying the number of gradient accumulation steps.
- `eos_token_id`: an integer specifying the end-of-sequence token ID.
- `lora_r`: an integer specifying the LORA parameter r.
- `lora_alpha`: an integer specifying the LORA parameter alpha.
- `lora_dropout`: a float specifying the LORA dropout rate.
- `learning_rate`: a float specifying the learning rate.
- `lr_scheduler_type`: a string specifying the type of learning rate scheduler.
- `num_warmup_steps`: an integer specifying the number of warmup steps.
- `weight_decay`: a float specifying the weight decay.
- `num_train_epochs`: an integer specifying the number of training epochs.
- `local-rank`: an integer specifying the local rank.
- `no_fp16`: a boolean flag indicating whether to use mixed precision or not.
- `bf16`: a boolean flag indicating whether to use bfloat16 or not.
- `no_gradient_checkpointing`: a boolean flag indicating whether to use gradient checkpointing or not.
- `seed`: an integer specifying the random seed.
- `num_workers`: an integer specifying the number of workers.
- `output_dir`: a string specifying the output directory.
- `log_freq`: an integer specifying the logging frequency.
- `delete_whitespaces`: a boolean flag indicating whether to delete whitespaces or not.
- `ignore_large_functions`: a boolean flag indicating whether to ignore large functions or not.
- `several_funcs_in_batch`: a boolean flag indicating whether to use several functions in a batch or not.
- `debug_on_small_model`: a boolean flag indicating whether to use debug mode on a small model or not.
- `max_funcs_in_seq`: an integer specifying the maximum number of functions in a sequence.
- `use_adalora`: a boolean flag indicating whether to use AdaLORA or not.
- `use_focal_loss`: a boolean flag indicating whether to use focal loss or not.
- `focal_loss_gamma`: a float specifying the focal loss gamma.
- `loss_reduction`: a string specifying the loss reduction method.
- `run_test`: a boolean flag indicating whether to run the test or not.
- `run_test_peft`: a boolean flag indicating whether to run the test PEFT or not.
- `model_checkpoint_path`: a string specifying the path to the model checkpoint.
- `use_vul_weights`: a boolean flag indicating whether to use vulnerability weights or not.
- `vul_weight`: a float specifying the vulnerability weight.
- `top_mlp_layers_num`: an integer specifying the number of top MLP layers.

An example of launching:
```
pip3 install accelerate einops bitsandbytes peft && export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' && python3 -m torch.distributed.launch --nproc_per_node 8 /home/ma-user/modelarts/inputs/code_1/finetune/run.py --dataset_tar_gz='/home/ma-user/modelarts/inputs/data_0/java_k_1_strict_2023_06_30.tar.gz' --split="train" --seq_length 2048 --batch_size 1 --gradient_accumulation_steps 4 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 1 --weight_decay 0.05 --output_dir='/home/ma-user/modelarts/outputs/results_0/' --log_freq=1 --delete_whitespaces --base_model starcoder --lora_r 8 --several_funcs_in_batch --num_train_epochs 50 --use_vul_weights --vul_weight 3.0 --use_focal_loss --focal_loss_gamma 1.0 --top_mlp_layers_num 0
```


# Running Docker Container and Script Tutorial

This tutorial will guide you through the steps required to run a Docker container locally, execute a script inside the container, and provide explanations for each step. The instructions assume that you have Docker installed on your local machine.

## Step 1: Building and Loading the Docker Image

To begin, build a Docker image named `sec-llm` from the provided Dockerfile `llm.Dockerfile` by executing the following command:
```
sudo docker build -f llm.Dockerfile -t sec-llm .
```

Next, save the built Docker image using the `docker save` command:
```
sudo docker save -o sec-llm.tar sec-llm
```

Finally, load the image using the `docker load` command. Make sure you navigate to the directory where the `sec-llm.tar` file is located, and then run the following command:

```
sudo docker load -i sec-llm.tar
```

## Step 2: Creating a Directory for Wheels
Next, create a directory called `wheels` to store the Python package wheels. Use the following command:

```
mkdir wheels
```
This will create a directory named `wheels` in the current working directory.

## Step 3: Installing Python Packages

Now, you need to install the required Python packages inside the Docker container. Use the pip wheel command with the -r flag to specify the requirements file and the --wheel-dir flag to specify the directory where the wheels should be stored. Run the following command:

```
pip wheel -r requirements.txt --wheel-dir=wheels
```
This command will install the packages specified in the `requirements.txt` file and save their wheels in the wheels directory.

## Step 4: Running the Docker Container

To run the Docker container locally in bash mode and execute the script inside it, you need to provide the appropriate volume mounts and run the container with the necessary configuration. Use the following command:
```
sudo docker run -u 0 --entrypoint /bin/bash -it -v /path/to/wheels:/wheels -v /path/to/data:/home/ma-user/modelarts/inputs/data_0 -v /path/to/code:/home/ma-user/modelarts/inputs/code_1 -v /path/to/model:/home/ma-user/modelarts/inputs/model_2 -v /path/to/outputs:/home/ma-user/modelarts/outputs -v /path/to/results:/home/ma-user/modelarts/outputs/results_0/ sec-llm
```

Make sure to replace `/path/to/wheels` with the actual path to the wheels directory created in Step 2. Similarly, replace the `/path/to/data`, `/path/to/code`, `/path/to/model`, `/path/to/outputs`, and `/path/to/results` with the respective paths where your data, code, model, outputs, and results directories are located.

This command will start the Docker container with the specified volume mounts, allowing data exchange between the container and your local machine.

## Step 5: Executing the Script inside the Docker Container

Once inside the Docker container, you can install the required Python packages from the previously created wheels directory and execute the script. Follow these commands:
```
pip install accelerate einops bitsandbytes peft --no-index --find-links=/wheels
```
```
python3 /home/ma-user/modelarts/inputs/code_1/finetune/run.py --dataset_tar_gz='/home/ma-user/modelarts/inputs/data_0/java_k_1_strict_2023_07_03.tar.gz' --split="train" --seq_length 2048 --batch_size 1 --gradient_accumulation_steps 160 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 1 --weight_decay 0.05 --output_dir='/home/ma-user/modelarts/outputs/results_0/' --log_freq=1 --ignore_large_functions --delete_whitespaces --base_model starcoder --lora_r 8
```
The first command installs the Python packages using the wheels from the wheels directory. The second command runs the script `run.py` with the specified arguments.

Another way of running the script:
```
python3 /home/ma-user/modelarts/inputs/code_1/finetune/run.py --dataset_tar_gz='/home/ma-user/modelarts/inputs/data_0/for_s3.tar.gz' --split="train" --seq_length 50 --batch_size 1 --gradient_accumulation_steps 160 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 1 --weight_decay 0.05 --output_dir='/home/ma-user/modelarts/outputs/results_0/' --log_freq=1 --delete_whitespaces --base_model starcoder --lora_r 8 --debug_on_small_model --several_funcs_in_batch
```

## Conclusion

By following these steps, you should be able to run the Docker container locally and execute the desired script inside it. Make sure to replace the file paths and directory paths with your own specific locations.