import math
import sys
from transformers.debug_utils import DebugOption
from transformers.utils import is_sagemaker_mp_enabled
from transformers import TrainerState
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.trainer_utils import has_length, ShardedDDPOption


def _build_debug_param_to_name_mapping_our_debug(ddp_model, parameters):
    param_to_param_index = {
        parameters[i]: i for i in range(len(parameters))
    }
    param_set = set(parameters)
    param_index_to_param_fqn = {}
    for module_name, module in ddp_model.module.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fqn = f"{module_name}.{param_name}"
            # Bypass ignored parameters since those are not reduced by DDP
            # to begin with.
            if fqn not in ddp_model.parameters_to_ignore and param.requires_grad:
                if param not in param_set:
                    ddp_model._log_and_throw(
                        ValueError,
                        f"Param with name {fqn} found in module parameters, but not DDP parameters."
                        " This indicates a bug in DDP, please report an issue to PyTorch.",
                    )
                param_index = param_to_param_index[param]
                param_index_to_param_fqn[param_index] = fqn

    # Ensure we covered all parameters
    if len(param_set) != len(param_index_to_param_fqn):
        ddp_model._log_and_throw(
            ValueError,
            (
                "Expected param to name mapping to cover all parameters, but"
                f" got conflicting lengths: {len(param_set)} vs "
                f"{len(param_index_to_param_fqn)}. This indicates a bug in DDP"
                ", please report an issue to PyTorch."
            ),
        )

    return param_index_to_param_fqn


def debug_params(trainer, batch_size=1):
    resume_from_checkpoint = None
    trainer.accelerator.free_memory()
    trainer._train_batch_size = batch_size
    # Data loader and number of training steps
    train_dataloader = trainer.get_train_dataloader()

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    args = trainer.args
    total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

    len_dataloader = None
    if has_length(train_dataloader):
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = trainer.num_examples(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = trainer.num_examples(train_dataloader) * args.num_train_epochs
    elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )

    # Compute absolute values for logging, eval, and save if given as ratio
    if args.logging_steps and args.logging_steps < 1:
        args.logging_steps = math.ceil(max_steps * args.logging_steps)
    if args.eval_steps and args.eval_steps < 1:
        args.eval_steps = math.ceil(max_steps * args.eval_steps)
    if args.save_steps and args.save_steps < 1:
        args.save_steps = math.ceil(max_steps * args.save_steps)

    if DebugOption.UNDERFLOW_OVERFLOW in trainer.args.debug:
        if trainer.args.n_gpu > 1:
            # nn.DataParallel(model) replicates the model, creating new variables and module
            # references registered here no longer work on other gpus, breaking the module
            raise ValueError(
                "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                " (torch.distributed.launch)."
            )
        else:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = (
            trainer.sharded_ddp is not None
            and trainer.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or trainer.fsdp is not None
    )

    if trainer.is_deepspeed_enabled:
        trainer.optimizer, trainer.lr_scheduler = deepspeed_init(trainer, num_training_steps=max_steps)

    if not delay_optimizer_creation:
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)

    trainer.state = TrainerState()
    trainer.state.is_hyper_param_search = False

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        trainer.model.gradient_checkpointing_enable()

    model = trainer._wrap_model(trainer.model_wrapped)

    if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
        trainer._load_from_checkpoint(resume_from_checkpoint, model)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is trainer.model else False

    if delay_optimizer_creation:
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        if hasattr(trainer.lr_scheduler, "step"):
            if trainer.use_apex:
                model = trainer.accelerator.prepare(trainer.model)
            else:
                model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                trainer.model, trainer.optimizer, trainer.lr_scheduler
            )

    if trainer.is_fsdp_enabled:
        trainer.model = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not trainer.model:
        trainer.model_wrapped = model

    # backward compatibility
    if trainer.is_deepspeed_enabled:
        trainer.deepspeed = trainer.model_wrapped

    # deepspeed ckpt loading
    if resume_from_checkpoint is not None and trainer.is_deepspeed_enabled:
        deepspeed_load_checkpoint(trainer.model_wrapped, resume_from_checkpoint)

    # Check if saved optimizer or scheduler states exist
    trainer._load_optimizer_and_scheduler(resume_from_checkpoint)

    parameters, expect_sparse_gradient = model._build_params_for_reducer()
    # print(parameters)
    param_to_name_mapping = _build_debug_param_to_name_mapping_our_debug(model, parameters)
    print(param_to_name_mapping)
    print(param_to_name_mapping[321])
    """
    param_list = list(model.named_parameters())
    param_name, param_321 = param_list[321]

    print(param_name)
    # print(param_list)
    """
