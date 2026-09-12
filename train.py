import sys
import importlib
from types import ModuleType

original_find_spec = importlib.util.find_spec
def mock_find_spec(name, package=None):
    if name == "peft" or name.startswith("peft."):
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = mock_find_spec

if "peft" in sys.modules:
    del sys.modules["peft"]

import os
os.environ["WANDB_MODE"] = "disabled"

import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import argparse
import itertools
import json
import logging
import math
import os
import shutil
import warnings
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from packaging import version
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor

import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):

    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import diffusers

class DeepSpeedModelWrapper(torch.nn.Module):

    def __init__(self, unet, image_encoder):
        super().__init__()
        self.unet = unet
        self.image_encoder = image_encoder

    def parameters(self, recurse: bool = True):

        for param in self.unet.parameters(recurse):
            yield param
        for param in self.image_encoder.parameters(recurse):
            yield param

class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=4, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_features, bias=False)

        torch.nn.init.normal_(self.lora_A.weight, std=0.02)
        torch.nn.init.zeros_(self.lora_B.weight)

        self.original_forward = None

    def forward(self, x):
        result = self.original_forward(x)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.scaling
        return result + lora_out

def add_lora_to_unet(unet, rank=4, target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                                                    "attn2.processor.to_k_ip", "attn2.processor.to_v_ip"]):

    lora_params = []
    for name, module in unet.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, torch.nn.Linear):

                lora = LoRALayer(module.in_features, module.out_features, rank=rank, alpha=rank).to(
                    device=module.weight.device, dtype=module.weight.dtype)
                lora.original_forward = module.forward

                module.forward = lora.forward

                module.lora = lora

                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

                lora_params.extend(list(lora.parameters()))
    return lora_params

def get_lora_state_dict(unet):

    state_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, 'lora'):
            lora = module.lora

            state_dict[f"unet.{name}.lora_A.weight"] = lora.lora_A.weight.data.half().clone()
            state_dict[f"unet.{name}.lora_B.weight"] = lora.lora_B.weight.data.half().clone()

            diffusers_name = name.replace(".", "_")
            state_dict[f"unet.{diffusers_name}.lora.up.weight"] = lora.lora_B.weight.data.half().clone()
            state_dict[f"unet.{diffusers_name}.lora.down.weight"] = lora.lora_A.weight.data.half().clone()
    return state_dict

def moving_average(data, window_size=100):

    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss_curve(loss_history, output_dir, window_size=100):

    if not loss_history:
        logger.warning("Loss history is empty; cannot plot the loss curve")
        return

    plt.figure(figsize=(12, 6))
    steps = range(1, len(loss_history) + 1)

    plt.plot(steps, loss_history, alpha=0.3, label='Average loss per step', color='#1f77b4')

    if len(loss_history) >= window_size:
        ma_loss = moving_average(loss_history, window_size)
        ma_steps = range(window_size, len(loss_history) + 1)
        plt.plot(ma_steps, ma_loss, label=f'Moving average (window={window_size} steps)', color='#ff7f0e', linewidth=2)

    plt.xlabel('Training steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training loss curve', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Loss curve saved to: {save_path}")

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler

def cast_training_params(model, dtype=torch.float32):

    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(dtype)

def compute_snr(noise_scheduler, timesteps):

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])
    return (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2

from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from src.pipeline import Pipeline
from src.dataset.data import KeypointsDataset, SimpleSubjectDrivenDataset
from src.modules.image_encoder import ImageEncoder
from src.modules.attention_processor import MaskedIPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
from src.utils import get_phrase_idx

if is_wandb_available():
    import wandb

check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-msdiffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
        help="Flag to additionally generate final state dict in the Kohya format so that it becomes compatible with A111, Comfy, Kohya, etc.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Automatically scale the learning rate by the effective total batch size (batch size × gradient accumulation steps × number of GPUs) using the linear scaling rule. When enabled, changing gradient accumulation or batch size does not require manually changing the learning rate.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--attn_align_weight",
        type=float,
        default=0.05,
        help="Attention-to-bbox alignment loss weight: forces subject-phrase text attention into the corresponding bbox; "
        "addresses subjects clustering at the image center or becoming positionally fixed. Set to 0 to disable this loss. Recommended: 0.05–0.1.",
    )
    parser.add_argument(
        "--rel_attn_weight",
        type=float,
        default=0.0,
        help="RAGC relational geometry loss weight: directed mass + pairwise overlap + center offset, using annotated bbox-pair geometry for self-supervision; "
        "constrains the pairwise geometry of subject-text attention to address relationship errors or positional fixation caused by attention targeting the wrong subject. Set to 0 to disable. Recommended: 0.02–0.05.",
    )
    parser.add_argument(
        "--rel_attn_lambda",
        type=float,
        default=0.15,
        help="Fixed RAGC directed-mass threshold λ: only a fallback when dir_area_adaptive=True (not used; instead "
        "λ=max(area(b_pa)+0.05, 0.10) is area-adaptive and enabled by default); when dir_area_adaptive=False, "
        "uses this fixed λ: a λ fraction of the agent attention mass must fall inside the patient box (for directed actions such as hug or point).",
    )
    parser.add_argument(
        "--rel_attn_dir_adaptive",
        type=lambda s: s.lower() in ("1", "true", "yes", "on"),
        default=True,
        help="Area-adaptive switch for the directed threshold (ablation 0-1): True → λ=max(area(b_pa)+γ, τ₀); "
        "False → fixed --rel_attn_lambda. Use with --rel_attn_ovl_symmetric for modified-vs-unmodified ablation.",
    )
    parser.add_argument(
        "--rel_attn_ovl_symmetric",
        type=lambda s: s.lower() in ("1", "true", "yes", "on"),
        default=True,
        help="Overlap-symmetrization switch (ablation 0-2): True → |IoU_annot−SoftIoU_attn| (penalizes both excessive proximity and distance); "
        "False → one-way ReLU(IoU_annot−SoftIoU_attn) (penalizes only excessive distance).",
    )
    parser.add_argument(
        "--clip_i_weight",
        type=float,
        default=0.0,
        help="CLIP-I identity-fidelity loss weight: reconstruct x0 → VAE-decode the bbox region → align CLIP features with the reference image (maximize cosine similarity); "
        "addresses generated subjects not resembling the reference. Set to 0 to disable. Recommended: 0.05–0.2.",
    )
    parser.add_argument(
        "--clip_i_max_t",
        type=int,
        default=300,
        help="Maximum timestep for CLIP-I loss (on a 1000-step scale): x0 reconstructed at high-noise steps is highly inaccurate, so compute only within this threshold.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. "
            "Set to 0 if multiprocessing loading is unstable; single-process loading is more compatible"
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Wether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://arxiv.org/abs/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )
    parser.add_argument(
        "--use_cross_attention",
        action="store_true",
        default=False,
        help="Whether to replace the original cosine-similarity adjacency matrix with the noise-subject cross-attention module",
    )
    parser.add_argument(
        "--cross_attention_heads",
        type=int,
        default=8,
        help="Number of cross-attention heads; effective only when use_cross_attention=True",
    )
    parser.add_argument(
        "--bidirectional_attention",
        action="store_true",
        help="Whether to use bidirectional cross-attention (noise and subjects learn associations from each other); effective only when use_cross_attention=True",
    )
    parser.add_argument(
        "--no-bidirectional_attention",
        dest="bidirectional_attention",
        action="store_false",
        help="Disable bidirectional cross-attention and use unidirectional mode",
    )
    parser.add_argument(
        "--enable_intra_type_attention",
        action="store_true",
        help="Whether to enable same-type intra-attention (within noise and subjects); enabled by default",
    )
    parser.add_argument(
        "--no-enable_intra_type_attention",
        dest="enable_intra_type_attention",
        action="store_false",
        help="Disable same-type intra-attention",
    )
    parser.add_argument(
        "--intra_attention_type",
        type=str,
        default="local_window",
        choices=["local_window", "axial", "global", "conv"],
        help="Noise intra-attention type: local_window (3×3 local window) / axial (axial attention) / global (global attention) / conv (depthwise separable convolution)",
    )
    parser.set_defaults(bidirectional_attention=True, enable_intra_type_attention=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:

        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args

def set_ms_adapter(unet: UNet2DConditionModel, scale=0.6, weight_dtype=torch.float16, num_tokens=16, text_tokens=77):

    attn_procs = {}
    attn_store = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                text_tokens=text_tokens,
                attn_store=attn_store,
                place_in_unet=name,
                need_text_attention_map=True
            ).to(unet.device, dtype=weight_dtype)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules, attn_store

def get_phrases_idx(tokenizer, phrases, prompt):

    res = []
    phrase_cnt = {}
    for phrase in phrases:
        if phrase in phrase_cnt:
            cur_cnt = phrase_cnt[phrase]
            phrase_cnt[phrase] += 1
        else:
            cur_cnt = 0
            phrase_cnt[phrase] = 1
        res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
    return res

def build_box_masks(boxes, h, w, dtype, device):

    b, n, _ = boxes.shape
    x_start = torch.floor(boxes[..., 0] * w)
    y_start = torch.floor(boxes[..., 1] * h)
    x_end = torch.ceil(boxes[..., 2] * w)
    y_end = torch.ceil(boxes[..., 3] * h)
    xs = torch.arange(w, device=device).view(1, 1, w).expand(b, n, w)
    ys = torch.arange(h, device=device).view(1, 1, h).expand(b, n, h)
    x_mask = (xs >= x_start.unsqueeze(-1)) & (xs < x_end.unsqueeze(-1))
    y_mask = (ys >= y_start.unsqueeze(-1)) & (ys < y_end.unsqueeze(-1))
    return (y_mask.unsqueeze(-1) & x_mask.unsqueeze(-2)).to(dtype)

def compute_attn_alignment_loss(attn_store, boxes, subject_phrase_idxes):

    subject_maps = {k: v for k, v in attn_store.items() if k.endswith("_subjects")}
    if not subject_maps:
        return None
    first_map = next(iter(subject_maps.values()))
    device, dtype = first_map.device, first_map.dtype
    boxes_dev = boxes.to(device=device, dtype=dtype)

    valid = ((subject_phrase_idxes[..., 0] > 0) &
             (subject_phrase_idxes[..., 1] > subject_phrase_idxes[..., 0])).to(device=device, dtype=dtype)
    if valid.sum() == 0:
        return None

    total_loss = 0.0
    num_layers = 0
    for key, attn_map in subject_maps.items():
        bsz, n, h, w = attn_map.shape
        if n != boxes_dev.shape[1]:
            continue
        box_masks = build_box_masks(boxes_dev, h, w, dtype, device)

        min_val = attn_map.flatten(2).min(dim=-1).values[..., None, None]
        max_val = attn_map.flatten(2).max(dim=-1).values[..., None, None]
        norm_map = (attn_map - min_val) / (max_val - min_val + 1e-5)

        per_subject = ((norm_map - box_masks) ** 2).flatten(2).mean(dim=-1)
        layer_loss = (per_subject * valid).sum() / valid.sum().clamp(min=1.0)
        total_loss = total_loss + layer_loss
        num_layers += 1

    if num_layers == 0:
        return None
    return total_loss / num_layers

import re as _re

_REL_PATTERNS = [
    (_re.compile(r'^A <(.+?)> and a <(.+?)> are (.+)$'), 'symmetric'),

    (_re.compile(r'^A <(.+?)> is to the (left|right) of a <(.+?)>$'), 'spatial'),
    (_re.compile(r'^A <(.+?)> is (above|below) a <(.+?)>$'), 'spatial'),
    (_re.compile(r'^A <(.+?)> is (behind|in front of) a <(.+?)>$'), 'spatial'),
    (_re.compile(r'^A <(.+?)> is (standing next to|sitting beside|lying next to|standing opposite|towering over) a <(.+?)>$'), 'spatial'),
    (_re.compile(r'^A <(.+?)> is (.+?) (?:a|an) <(.+?)>$'), 'directed'),
]

_nlp = None

def _get_nlp():

    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logging.getLogger(__name__).info("spaCy loaded; dependency parsing enabled as a fallback for RAGC relation extraction")
        except Exception as e:
            logging.getLogger(__name__).warning(f"spaCy unavailable; RAGC relation extraction will rely only on template regex: {e}")
            _nlp = False
    return _nlp or None

def _match_subject_idx(token_text, phrases):

    t = token_text.strip("<>").strip()
    for i, ph in enumerate(phrases):
        p = ph.strip("<>").strip()
        if t == p or t in p or p in t:
            return i
    return None

def extract_relation_info(prompts, phrases_list):

    infos = []
    for prompt, phrases in zip(prompts, phrases_list):
        info = None

        m_passive = _re.match(r'^(?:A\s+)?<([^>]+)>\s+is\s+.*?thrown by a <([^>]+)>$', prompt.strip())
        if m_passive:
            agent_tok = m_passive.group(1)
            by_tok = m_passive.group(2)
            ai = _match_subject_idx(agent_tok, phrases)
            bi = _match_subject_idx(by_tok, phrases)
            if ai is not None and bi is not None and ai != bi:

                info = {'type': 'directed', 'agent': bi, 'patient': ai}
        if info is None:
            for pat, typ in _REL_PATTERNS:
                m = pat.match(prompt.strip())
                if m:
                    if typ == 'directed':
                        ai = _match_subject_idx(m.group(1), phrases)
                        bi = _match_subject_idx(m.group(3), phrases)
                        if ai is not None and bi is not None and ai != bi:
                            info = {'type': 'directed', 'agent': ai, 'patient': bi}
                    elif typ == 'symmetric':
                        ai = _match_subject_idx(m.group(1), phrases)
                        bi = _match_subject_idx(m.group(2), phrases)
                        if ai is not None and bi is not None and ai != bi:
                            info = {'type': 'symmetric', 'a': ai, 'b': bi}
                    else:
                        ai = _match_subject_idx(m.group(1), phrases)
                        bi = _match_subject_idx(m.group(3), phrases)
                        if ai is not None and bi is not None and ai != bi:
                            info = {'type': 'spatial', 'pos': ai, 'ref': bi, 'kind': m.group(2)}
                    break

        if info is None:
            m = _re.match(r'^(?:A\s+)?<([^>]+)>\s+is\s+(.+)$', prompt.strip())
            if m:
                agent_tok = m.group(1)
                rest = m.group(2)

                m_by = _re.search(r'thrown by a <([^>]+)>', rest)
                if m_by:
                    patient_tok = m_by.group(1)
                    ai = _match_subject_idx(agent_tok, phrases)
                    bi = _match_subject_idx(patient_tok, phrases)
                    if ai is not None and bi is not None and ai != bi:

                        info = {'type': 'directed', 'agent': bi, 'patient': ai}
                else:
                    m_y = _re.search(r'<([^>]+)>', rest)
                    if m_y:
                        ai = _match_subject_idx(agent_tok, phrases)
                        bi = _match_subject_idx(m_y.group(1), phrases)
                        if ai is not None and bi is not None and ai != bi:
                            info = {'type': 'directed', 'agent': ai, 'patient': bi}

        if info is None:
            nlp = _get_nlp()
            if nlp is not None:
                try:
                    doc = nlp(prompt)
                    verb = next((t for t in doc if t.pos_ == 'VERB'), None)
                    if verb is not None:
                        subj = next((t for t in doc if t.dep_ in ('nsubj', 'nsubjpass')), None)
                        obj = next((t for t in doc if t.dep_ in ('dobj', 'attr', 'oprd')), None)
                        if subj is not None and obj is not None:
                            si = _match_subject_idx(subj.text, phrases)
                            oi = _match_subject_idx(obj.text, phrases)
                            if si is not None and oi is not None and si != oi:
                                info = {'type': 'directed', 'agent': si, 'patient': oi}
                except Exception as e:
                    logger.warning(f"spaCy relation parsing failed; skipping relational constraints for this sample: {e}")
        infos.append(info)
    return infos

def _box_iou(b1, b2):

    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    a1 = (b1[2] - b1[0]).clamp(min=0) * (b1[3] - b1[1]).clamp(min=0)
    a2 = (b2[2] - b2[0]).clamp(min=0) * (b2[3] - b2[1]).clamp(min=0)
    return inter / (a1 + a2 - inter + 1e-5)

def compute_relation_geometry_loss(attn_store, boxes, rel_infos, lambda_dir=0.15,
                                   dir_area_adaptive=True, dir_gamma=0.05, dir_tau0=0.10,
                                   ovl_symmetric=True):

    subject_maps = {k: v for k, v in attn_store.items() if k.endswith("_subjects")}
    if not subject_maps or boxes is None:
        return None
    device = boxes.device
    dirs, ovls, ords = [], [], []
    for _, attn in sorted(subject_maps.items()):
        b, n, h, w = attn.shape
        masks = build_box_masks(boxes, h, w, torch.float32, device)
        mn = attn.flatten(2).min(dim=-1).values[..., None, None]
        mx = attn.flatten(2).max(dim=-1).values[..., None, None]
        a = (attn - mn) / (mx - mn + 1e-5)
        for bi in range(b):
            info = rel_infos[bi]
            if info is None or n < 2:
                continue
            i, j = 0, 1
            ai, aj = a[bi, i], a[bi, j]

            if info.get('type') == 'directed':
                ag, pa = info['agent'], info['patient']

                mass_pa = (a[bi, ag].float() * masks[bi, pa].float()).sum() / (a[bi, ag].float().sum() + 1e-5)
                if dir_area_adaptive:

                    b_pa = boxes[bi, pa]
                    area_pa = ((b_pa[2] - b_pa[0]).clamp(min=0) * (b_pa[3] - b_pa[1]).clamp(min=0))
                    thr = (area_pa + dir_gamma).clamp(min=dir_tau0)
                else:
                    thr = lambda_dir
                dirs.append(F.relu(thr - mass_pa))

            ovl_attn = ai.float().minimum(aj.float()).sum() / (ai.float().maximum(aj.float()).sum() + 1e-5)
            ovl_annot = _box_iou(boxes[bi, i], boxes[bi, j])
            if ovl_symmetric:

                ovls.append((ovl_annot - ovl_attn).abs())
            else:
                ovls.append(F.relu(ovl_annot - ovl_attn))

            def _centroid(m):

                tot = m.sum().float() + 1e-5
                mf = m.float()
                ys = torch.arange(h, device=device).float().view(h, 1) * (mf / tot)
                xs = torch.arange(w, device=device).float().view(1, w) * (mf / tot)
                return xs.sum() / w, ys.sum() / h
            cx_i, cy_i = _centroid(ai); cx_j, cy_j = _centroid(aj)
            bxi = boxes[bi, i]; bxj = boxes[bi, j]
            cxi = (bxi[0] + bxi[2]) / 2; cyi = (bxi[1] + bxi[3]) / 2
            cxj = (bxj[0] + bxj[2]) / 2; cyj = (bxj[1] + bxj[3]) / 2
            ords.append((cx_i - cx_j - (cxi - cxj)).abs() + (cy_i - cy_j - (cyi - cyj)).abs())
    if not (dirs or ovls or ords):
        return None
    out = {}
    if dirs:
        out['dir'] = torch.stack(dirs).mean()
    if ovls:
        out['ovl'] = torch.stack(ovls).mean()
    if ords:
        out['ord'] = torch.stack(ords).mean()
    return out

def compute_clip_identity_loss(model_input, model_pred, timesteps, vae, clip_model,
                               concept_images, boxes, subject_phrase_idxes,
                               alphas_cumprod, max_t=300, crop_margin=8,
                               max_crop_latent=32):

    b, n, _ = boxes.shape
    if n == 0 or vae is None or clip_model is None:
        return None
    valid_t = timesteps < max_t
    if not valid_t.any():
        return None

    phrase_valid = ((subject_phrase_idxes[..., 0] != 0) | (subject_phrase_idxes[..., 1] != 0)).to(model_input.device)
    bbox_valid = ((boxes[..., 2] - boxes[..., 0] > 0.02) & (boxes[..., 3] - boxes[..., 1] > 0.02)).to(model_input.device)
    valid = phrase_valid & bbox_valid
    if not valid.any():
        return None

    alpha_bar = alphas_cumprod.to(model_input.device)[timesteps].view(-1, 1, 1, 1).to(model_input.dtype)

    x0 = (model_input - (1 - alpha_bar).sqrt() * model_pred) / alpha_bar.sqrt()

    clip_dtype = next(clip_model.parameters()).dtype
    with torch.no_grad():
        ref_flat = concept_images.flatten(0, 1).to(model_input.device).to(dtype=clip_dtype)
        ref_feats = clip_model(ref_flat).image_embeds.float()
        ref_feats = F.normalize(ref_feats, dim=-1)
    vae_scale = vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") else 0.18215

    gen_pixels = []
    valid_pairs = []
    H, W = model_input.shape[-2:]
    for i in range(b):
        if not valid_t[i]:
            continue
        for j in range(n):
            if not valid[i, j]:
                continue
            x1, y1, x2, y2 = boxes[i, j]
            cw = min(int(x2 * W) - int(x1 * W), max_crop_latent)
            ch = min(int(y2 * H) - int(y1 * H), max_crop_latent)
            if cw < 8 or ch < 8:
                continue
            px1 = max(0, min(int(x1 * W) - crop_margin, W - cw))
            py1 = max(0, min(int(y1 * H) - crop_margin, H - ch))
            cw = (cw // 8) * 8; ch = (ch // 8) * 8
            if cw < 8 or ch < 8:
                continue
            crop = x0[i, :, py1:py1 + ch, px1:px1 + cw]
            crop_float = crop.float() / vae_scale
            dec = vae.decode(crop_float.unsqueeze(0)).sample
            dec = (dec.float() + 1.0) / 2.0

            processed = F.interpolate(dec, size=(224, 224), mode='bilinear', align_corners=False)
            _mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=processed.device).view(1, 3, 1, 1)
            _std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=processed.device).view(1, 3, 1, 1)
            processed = (processed - _mean) / _std
            gen_pixels.append(processed)
            valid_pairs.append((i, j))
    if not gen_pixels:
        return None
    gen_imgs = torch.cat(gen_pixels, dim=0).to(clip_dtype)

    gen_feats = torch.utils.checkpoint.checkpoint(
        lambda m, x: m(x).image_embeds, clip_model, gen_imgs, use_reentrant=False,
    ).float()
    gen_feats = F.normalize(gen_feats, dim=-1)
    refs = torch.stack([ref_feats[i * n + j] for (i, j) in valid_pairs])
    return (1.0 - (gen_feats * refs).sum(dim=-1)).mean()

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def collate_fn(examples):
    pixel_values = [example["images"] for example in examples]
    concept_images = [example["concept_images"] for example in examples]
    prompts = [example["prompts"] for example in examples]
    entities = [example["entities"] for example in examples]
    predicates = [example["predicate"] for example in examples]
    bboxes = [example["bboxes"] for example in examples]
    object_segmaps = [example["object_segmaps"] for example in examples]
    keypoints = [example["keypoints"] for example in examples]
    concept_images_for_vae = [example["concept_images_for_vae"] for example in examples]
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    concept_images = torch.stack(concept_images)
    bboxes = torch.stack(bboxes)
    object_segmaps = torch.stack(object_segmaps)
    keypoints = torch.stack(keypoints)
    concept_images_for_vae = torch.stack(concept_images_for_vae)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    concept_images = concept_images.to(memory_format=torch.contiguous_format).float()
    bboxes = bboxes.to(memory_format=torch.contiguous_format).float()

    batch = {
        "images": pixel_values,
        "concept_images": concept_images,
        "prompts": prompts,
        "bboxes": bboxes,
        "object_segmaps": object_segmaps,
        "keypoints": keypoints,
        "concept_images_for_vae": concept_images_for_vae,
        "entities": entities,
        "predicates": predicates,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.do_edm_style_training and args.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":

        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    logger.info("🚀 Loading model components. SDXL and CLIP total approximately 20 GB; the first load may take 2–5 minutes. Please wait...")

    logger.info("  Loading tokenizer...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    scheduler_type = determine_scheduler_type(args.pretrained_model_name_or_path, args.revision)
    if "EDM" in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("Performing EDM-style training!")
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    ms_ckpt = "../checkpoints/MS-Diffusion/ms_adapter.bin"
    ms_state_dict = torch.load(ms_ckpt)
    image_encoder_state_dict = {}
    for key, value in ms_state_dict.items():
        if key == "image_proj":
            for k, v in value.items():
                image_encoder_state_dict["resampler." + k] = v
        elif key == "dummy_image_tokens":
            image_encoder_state_dict[key] = value
        else:
            ms_adapter_state_dict = value

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    adapter_modules, attn_store = set_ms_adapter(unet, scale=0.6)
    adapter_modules.load_state_dict(ms_adapter_state_dict)

    clip_model_name_or_path = "../laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    logger.info("  Loading the CLIP vision encoder (approximately 10 GB)...")
    image_encoder = ImageEncoder(
        clip_model_name_or_path,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
        latent_init_mode="grounding",
        phrase_embeddings_dim=text_encoder_one.config.projection_dim,
        gcn_num_layers=2,
        use_cross_attention=args.use_cross_attention,
        cross_attention_heads=args.cross_attention_heads,
        bidirectional_attention=args.bidirectional_attention,
        enable_intra_type_attention=args.enable_intra_type_attention,
        intra_attention_type=args.intra_attention_type,
    )
    image_encoder.load_state_dict(image_encoder_state_dict, strict=False)

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)

    for name, param in image_encoder.named_parameters():
        if "gcn_layers" in name or "noise_proj" in name or "noise_gcn_layers" in name or "noise_reproj" in name or "noise_output_norm" in name or "noise_fusion_net" in name or "fine_grained_fusion_net" in name or "subject_fusion_net" in name or "cross_attention" in name:
            param.requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:

        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    unet.to(accelerator.device, dtype=weight_dtype)

    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    image_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    unet_lora_parameters = add_lora_to_unet(
        unet,
        rank=args.rank,
    )
    logger.info(f"✅ Number of UNet LoRA parameters:{sum(p.numel() for p in unet_lora_parameters)/1e6:.2f}M(cross-attention layers only)")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, type(unwrap_model(image_encoder))):

                    gcn_state = {
                        k:v for k,v in model.state_dict().items()
                        if "gcn_layers" in k or "noise_proj" in k or "noise_gcn_layers" in k or "noise_reproj" in k or "noise_output_norm" in k or "noise_fusion_net" in k or "fine_grained_fusion_net" in k or "subject_fusion_net" in k or "cross_attention" in k
                    }
                    torch.save(gcn_state, os.path.join(output_dir, "gcn_weights.pt"))
                    logger.info(f"All GCN-related weights saved to {output_dir}/gcn_weights.pt")
                elif isinstance(model, type(unwrap_model(unet))):

                    unet_lora_layers = get_lora_state_dict(model)
                    Pipeline.save_lora_weights(
                        output_dir,
                        unet_lora_layers=unet_lora_layers,
                        text_encoder_lora_layers=None,
                        text_encoder_2_lora_layers=None,
                    )
                    logger.info(f"UNet LoRA weights saved to {output_dir}/pytorch_lora_weights.safetensors(cross-attention layers only)")

        for _ in range(len(models)):
            weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(image_encoder))):

                if os.path.exists(os.path.join(input_dir, "gcn_weights.pt")):
                    gcn_state = torch.load(os.path.join(input_dir, "gcn_weights.pt"))
                    model.load_state_dict(gcn_state, strict=False)
                    logger.info(f"All GCN-related weights loaded from {input_dir}/gcn_weights.pt loaded")
            elif isinstance(model, type(unwrap_model(unet))):

                pass

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        scale_factor = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        original_lr = args.learning_rate
        original_text_encoder_lr = args.text_encoder_lr

        args.learning_rate = args.learning_rate * scale_factor
        args.text_encoder_lr = args.text_encoder_lr * scale_factor

        logger.info(f"✅ Automatic learning-rate scaling enabled; scale factor:{scale_factor}")
        logger.info(f"  Original UNet/GCN learning rate:{original_lr:.2e} → scaled:{args.learning_rate:.2e}")
        logger.info(f"  Original text-encoder learning rate:{original_text_encoder_lr:.2e} → scaled:{args.text_encoder_lr:.2e}")

        if args.learning_rate > 1e-3:
            logger.warning(f"⚠️  Scaled UNet/GCN learning rate({args.learning_rate:.2e})is too high and may destabilize training; consider lowering the base learning rate.")
        if args.text_encoder_lr > 1e-4:
            logger.warning(f"⚠️  Scaled text-encoder learning rate({args.text_encoder_lr:.2e})is too high and may cause semantic collapse; consider lowering the base learning rate.")

    if args.mixed_precision == "fp16":
        models = [unet, image_encoder]

        cast_training_params(models, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    gcn_parameters = list(filter(lambda p: p.requires_grad, image_encoder.parameters()))

    total_trainable = sum(p.numel() for p in unet_lora_parameters) + sum(p.numel() for p in gcn_parameters)

    original_gcn_params = []
    noise_gcn_params = []
    fusion_net_params = []
    fine_grained_fusion_params = []
    subject_fusion_params = []
    cross_attention_params = []
    for n, p in image_encoder.named_parameters():
        if p.requires_grad:
            if "gcn_layers" in n and not "noise_gcn" in n:
                original_gcn_params.append(p)
            elif "noise_proj" in n or "noise_gcn_layers" in n or "noise_reproj" in n or "noise_output_norm" in n:
                noise_gcn_params.append(p)
            elif "noise_fusion_net" in n:
                fusion_net_params.append(p)
            elif "fine_grained_fusion_net" in n:
                fine_grained_fusion_params.append(p)
            elif "subject_fusion_net" in n:
                subject_fusion_params.append(p)
            elif "cross_attention" in n:
                cross_attention_params.append(p)

    unet_self_attn_lora_params = []
    unet_cross_attn_lora_params = []
    for n, p in unet.named_parameters():
        if p.requires_grad and "lora" in n:
            if "attn1" in n:
                unet_self_attn_lora_params.append(p)
            elif "attn2" in n:
                unet_cross_attn_lora_params.append(p)

    subject_gcn_total = sum(p.numel() for p in original_gcn_params)

    joint_gcn_total = sum(p.numel() for p in noise_gcn_params) +\
                      sum(p.numel() for p in fusion_net_params) +\
                      sum(p.numel() for p in fine_grained_fusion_params) +\
                      sum(p.numel() for p in subject_fusion_params) +\
                      sum(p.numel() for p in cross_attention_params)

    unet_lora_total = sum(p.numel() for p in unet_cross_attn_lora_params)

    logger.info(f"✅ Total trainable parameters:{total_trainable/1e6:.2f}M")
    logger.info(f"  🧠 (1)Subject-relation GCN: {subject_gcn_total/1e6:.2f}M")
    logger.info(f"    Modules: subject token-level GCN layers (gcn_layers)")
    logger.info(f"  🔗 (2)Joint subject-noise GCN: {joint_gcn_total/1e6:.2f}M")
    logger.info(f"    Modules: noise projection (noise_proj) + joint GCN layers (noise_gcn_layers) + noise reprojection (noise_reproj) + joint GCN output normalization (noise_output_norm) + adaptive noise fusion network (noise_fusion_net) + fine-grained multiscale fusion network (fine_grained_fusion_net) + subject-node fusion network (subject_fusion_net) + noise-subject cross-attention module (cross_attention)")
    logger.info(f"  🎨 (3)UNet fine-tuning parameters: {unet_lora_total/1e6:.2f}M")
    logger.info(f"    Modules: LoRA low-rank parameters for UNet cross-attention layers (attn2)")

    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    gcn_parameters_with_lr = {"params": gcn_parameters, "lr": args.learning_rate}
    params_to_optimize = [gcn_parameters_with_lr, unet_lora_parameters_with_lr]

    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    logger.info("📂 Loading training dataset...")
    train_dataset = SimpleSubjectDrivenDataset(root=args.instance_data_dir)
    logger.info(f"✅ Dataset loaded; total {len(train_dataset)} training samples")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
    )

    def compute_time_ids(original_size, crops_coords_top_left):

        target_size = (args.resolution, args.resolution)
        add_time_ids = list(tuple(original_size) + tuple(crops_coords_top_left) + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model_wrapper = DeepSpeedModelWrapper(unet, image_encoder)

    model_wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model_wrapper, optimizer, train_dataloader, lr_scheduler
    )

    unet = model_wrapper.unet
    image_encoder = model_wrapper.image_encoder

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = (
            "msdiffusion-gcn-lora-sd-xl"
            if "playground" not in args.pretrained_model_name_or_path
            else "msdiffusion-gcn-lora-playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:

            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            checkpoint_path = os.path.join(args.output_dir, path)
            if accelerator.is_main_process:

                if os.path.exists(os.path.join(checkpoint_path, "gcn_weights.pt")):
                    gcn_state = torch.load(os.path.join(checkpoint_path, "gcn_weights.pt"))
                    unwrapped_encoder = unwrap_model(image_encoder)
                    unwrapped_encoder.load_state_dict(gcn_state, strict=False)
                    logger.info(f"GCN weights loaded from {checkpoint_path}/gcn_weights.pt loaded")

                if os.path.exists(os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")):
                    from safetensors.torch import load_file
                    lora_state = load_file(os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors"))

                    logger.info(f"UNet LoRA weights loaded from {checkpoint_path}/pytorch_lora_weights.safetensors loaded")

                if os.path.exists(os.path.join(checkpoint_path, "train_state.pt")):
                    train_state = torch.load(os.path.join(checkpoint_path, "train_state.pt"))
                    global_step = train_state["global_step"]
                    epoch = train_state["epoch"]
                    loss_history = train_state["loss_history"]
                    optimizer.load_state_dict(train_state["optimizer_state"])
                    lr_scheduler.load_state_dict(train_state["lr_scheduler_state"])
                    logger.info(f"Training state loaded from {checkpoint_path}/train_state.pt loaded")

            global_step = torch.tensor(global_step, device=accelerator.device)
            torch.distributed.broadcast(global_step, src=0)
            global_step = global_step.item()

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",

        disable=not accelerator.is_local_main_process,
    )

    loss_history = []
    current_accum_losses = []

    record_loss = accelerator.is_main_process

    loss_log_path = os.path.join(args.output_dir, 'loss_log.csv') if record_loss else None

    if record_loss:

        if args.resume_from_checkpoint and os.path.exists(loss_log_path):

            with open(loss_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
            loss_history = []
            for line in lines:
                if line.strip():
                    _, loss = line.strip().split(',')
                    loss_history.append(float(loss))
            logger.info(f"Historical loss records loaded; total {len(loss_history)} steps")
        else:

            with open(loss_log_path, 'w', encoding='utf-8') as f:
                f.write("step,loss\n")
            logger.info(f"Loss log file created: {loss_log_path}")

    logger.info("🚀 Initialization complete; preparing to enter the training loop...")
    logger.info("📦 Loading the first training batch. The first load may take 1–3 minutes depending on dataset size and disk speed. Please wait...")
    logger.info("💡 If there is still no response after 5 minutes, check: 1. the dataset path; 2. whether dataset samples are corrupted; 3. try setting --dataloader_num_workers to 0")

    logger.info("🔍 Debug: starting dataset-loading test...")
    try:

        first_batch = next(iter(train_dataloader))
        logger.info(f"✅ Dataset-loading test succeeded; image shape of the first batch:{first_batch['images'].shape}")
    except Exception as e:
        logger.error(f"❌ Dataset loading failed; error:{str(e)}")
        raise e

    logger.info("🔍 Debug: dataset loading is normal; preparing to enter the training loop...")

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        image_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["images"].to(dtype=vae.dtype)
                concept_images_for_vae = batch["concept_images_for_vae"].to(dtype=vae.dtype)
                prompts = batch["prompts"]
                concept_images = batch["concept_images"]
                boxes = batch["bboxes"]
                object_segmaps = batch["object_segmaps"]
                phrases = batch["entities"]
                keypoints = batch["keypoints"]
                predicates = batch["predicates"]
                phrase_idxes = []
                for predicate, prompt in zip(predicates, prompts):

                    phrase_idx, _ = get_phrase_idx(tokenizer_one, predicate, prompt)
                    phrase_idxes.append(phrase_idx)
                phrase_idxes = torch.tensor(phrase_idxes)

                subject_phrase_idxes = []
                for ents, prompt in zip(phrases, prompts):
                    subject_phrase_idxes.append(get_phrases_idx(tokenizer_one, ents, prompt))
                subject_phrase_idxes = torch.tensor(subject_phrase_idxes, dtype=torch.long)

                rel_infos = extract_relation_info(prompts, phrases)

                bsz, n, c, h, w = concept_images.shape
                gt_keypoints = keypoints[:, :n, ...]
                input_keypoints = keypoints[:, n:, ...]

                drop_grounding_tokens = [0] * bsz
                batch_boxes = boxes.view(bsz * boxes.shape[1], -1).to(accelerator.device, dtype=weight_dtype)

                phrase_input_ids = []
                for phrase in phrases:
                    phrase_input_id = tokenizer_one(
                        phrase,
                        max_length=tokenizer_one.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
                    phrase_input_ids.append(phrase_input_id)
                phrase_input_ids = torch.stack(phrase_input_ids, dim=0).to(accelerator.device)
                phrase_input_ids = phrase_input_ids.view(-1, phrase_input_ids.shape[-1])
                phrase_embeds = text_encoder_one(phrase_input_ids).pooler_output.to(dtype=weight_dtype)
                grounding_kwargs = {"boxes": batch_boxes, "phrase_embeds": phrase_embeds, "drop_grounding_tokens": drop_grounding_tokens}

                model_input = vae.encode(pixel_values).latent_dist.sample()
                concept_latents = vae.encode(concept_images_for_vae.flatten(0, 1)).latent_dist.sample()
                concept_latents = concept_latents * vae.config.scaling_factor
                concept_latents = concept_latents.reshape(bsz, -1, *model_input.shape[-3:])
                concept_latents = concept_latents.permute(0, 1, 3, 4, 2)

                if latents_mean is None and latents_std is None:
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                else:
                    latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                    latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                    model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                if not args.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()
                else:

                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                updated_noisy_input, image_prompt_embeds, _ = image_encoder(
                    concept_images.to(accelerator.device, dtype=weight_dtype),
                    None,
                    grounding_kwargs,
                    run_full_pipeline=True,
                    noisy_model_input=noisy_model_input
                )

                noisy_model_input = updated_noisy_input.to(dtype=weight_dtype)
                image_prompt_embeds = image_prompt_embeds.to(dtype=weight_dtype)

                if not args.train_text_encoder:
                    prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )
                    prompt_embeds_input = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts)

                if args.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                    if "EDM" in scheduler_type:
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                    else:
                        inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=s, crops_coords_top_left=c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )
                cross_attention_kwargs = {"boxes": boxes, "phrase_idxes": phrase_idxes,
                                           "subject_phrase_idxes": subject_phrase_idxes}

                if not args.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": unet_add_text_embeds,
                    }
                    model_pred = unet(
                        inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds}
                    )

                    prompt_embeds_input = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                    model_pred = unet(
                        inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                weighting = None
                if args.do_edm_style_training:

                    if "EDM" in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_model_input
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas ** 2 + 1) ** 0.5) + (
                                noisy_model_input / (sigmas ** 2 + 1)
                            )

                    if "EDM" not in scheduler_type:
                        weighting = (sigmas**-2.0).float()

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_input if args.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        model_input
                        if args.do_edm_style_training
                        else noise_scheduler.get_velocity(model_input, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                else:

                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":

                        mse_loss_weights = base_weight + 1
                    else:

                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if args.attn_align_weight > 0:
                    align_loss = compute_attn_alignment_loss(attn_store, boxes, subject_phrase_idxes)
                    if align_loss is not None:
                        loss = loss + args.attn_align_weight * align_loss.to(loss.dtype)

                if args.rel_attn_weight > 0:
                    rel_losses = compute_relation_geometry_loss(
                        attn_store, boxes, rel_infos,
                        lambda_dir=args.rel_attn_lambda,
                        dir_area_adaptive=args.rel_attn_dir_adaptive,
                        ovl_symmetric=args.rel_attn_ovl_symmetric,
                    )
                    if rel_losses is not None:
                        rel_tot = sum(rel_losses.values())
                        loss = loss + args.rel_attn_weight * rel_tot.to(loss.dtype)

                if args.clip_i_weight > 0:
                    clip_loss = compute_clip_identity_loss(
                        noisy_model_input, model_pred, timesteps, vae,
                        image_encoder.clip_model,
                        concept_images, boxes, subject_phrase_idxes,
                        noise_scheduler.alphas_cumprod,
                        max_t=args.clip_i_max_t,
                    )
                    if clip_loss is not None:
                        loss = loss + args.clip_i_weight * clip_loss.to(loss.dtype)

                accelerator.backward(loss)
                if accelerator.sync_gradients:

                    params_to_clip = list(gcn_parameters) + list(unet_lora_parameters)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                keys = list(attn_store.keys())
                for key in keys:
                    del attn_store[key]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:

                    if record_loss and current_accum_losses:
                        avg_loss = sum(current_accum_losses) / len(current_accum_losses)
                        loss_history.append(avg_loss)
                        current_accum_losses.clear()

                        with open(loss_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{global_step},{avg_loss:.6f}\n")
                            f.flush()

                if global_step % args.checkpointing_steps == 0:

                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    model_wrapper.save_checkpoint(
                        save_dir=ckpt_dir,
                        tag="global_step",
                        client_state={
                            "global_step": global_step,
                            "epoch": epoch,
                            "loss_history": loss_history,
                        },
                        save_latest=False,
                    )

                    if accelerator.is_main_process:

                        ckpt_path = os.path.join(ckpt_dir, "global_step")
                        unwrapped_encoder = unwrap_model(image_encoder)
                        gcn_keys = [
                            "gcn_layers", "noise_proj", "noise_gcn_layers", "noise_reproj",
                            "noise_output_norm", "noise_fusion_net", "fine_grained_fusion_net",
                            "subject_fusion_net", "cross_attention",
                        ]

                        gcn_state = {
                            k: v.detach().cpu().clone()
                            for k, v in unwrapped_encoder.state_dict().items()
                            if any(x in k for x in gcn_keys)
                        }

                        torch.save(gcn_state, os.path.join(ckpt_dir, "gcn_weights.pt"))
                        logger.info(f"GCN weights saved to {ckpt_dir}/gcn_weights.pt")

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_layers = get_lora_state_dict(unwrapped_unet)
                        from safetensors.torch import save_file
                        filtered_lora_state = {
                            k: v for k, v in unet_lora_layers.items()
                            if not ("_" in k and "lora.up" in k or "lora.down" in k)
                        }
                        save_file(filtered_lora_state, os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"))
                        logger.info(f"UNet LoRA weights saved to {ckpt_dir}/pytorch_lora_weights.safetensors")

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                for rc in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, rc))

                        if len(loss_history) >= 10:
                            plot_loss_curve(loss_history, args.output_dir, window_size=100)
                            logger.info(f"✅ Updated the latest loss curve (current step:{global_step})")

                        logger.info(f"✅ Intermediate checkpoint saved:{ckpt_dir}")

                    accelerator.wait_for_everyone()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if record_loss:
                current_accum_losses.append(logs["loss"])

            if global_step >= args.max_train_steps:

                if not accelerator.is_main_process:
                    import sys
                    sys.exit(0)
                break

    logger.info("🔧 Saving final weights (main process only; no distributed synchronization)...")

    last_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    if os.path.exists(last_checkpoint_dir):
        import shutil

        src_gcn = os.path.join(last_checkpoint_dir, "gcn_weights.pt")
        dst_gcn = os.path.join(args.output_dir, "final_gcn_weights.pt")
        if os.path.exists(src_gcn):
            shutil.copy(src_gcn, dst_gcn)
            logger.info(f"✅ All GCN-related weights copied to {dst_gcn}")

        src_lora = os.path.join(last_checkpoint_dir, "pytorch_lora_weights.safetensors")
        dst_lora = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(src_lora):
            shutil.copy(src_lora, dst_lora)
            logger.info(f"✅ Final UNet LoRA weights copied to {dst_lora}")

        if record_loss:
            plot_loss_curve(loss_history, args.output_dir, window_size=100)
            logger.info(f"✅ Final loss curve saved")
    else:

        final_ckpt_path = os.path.join(args.output_dir, "final_gcn_weights.pt")
        unwrapped_encoder = unwrap_model(image_encoder)

        full_encoder_state = accelerator.get_state_dict(image_encoder)
        gcn_state = {
            k:v for k,v in full_encoder_state.items()
            if "gcn_layers" in k or "noise_proj" in k or "noise_gcn_layers" in k or "noise_reproj" in k or "noise_output_norm" in k or "noise_fusion_net" in k or "fine_grained_fusion_net" in k or "subject_fusion_net" in k or "cross_attention" in k
        }
        torch.save(gcn_state, final_ckpt_path)
        logger.info(f"✅ All GCN-related weights saved to {final_ckpt_path}")

        unwrapped_unet = unwrap_model(unet)
        unet_lora_layers = get_lora_state_dict(unwrapped_unet)
        from safetensors.torch import save_file
        filtered_lora_state = {}
        for k, v in unet_lora_layers.items():
            if not ("_" in k and "lora.up" in k or "lora.down" in k):
                filtered_lora_state[k] = v
        save_file(filtered_lora_state, os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"))
        logger.info(f"✅ Final UNet LoRA weights saved to {args.output_dir}/pytorch_lora_weights.safetensors")

        if record_loss:
            plot_loss_curve(loss_history, args.output_dir, window_size=100)
            logger.info(f"✅ Final loss curve saved")

    logger.info("🎉 Training complete. All weights were saved successfully; exiting normally")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    main(args)
