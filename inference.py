import sys
import importlib

original_find_spec = importlib.util.find_spec
def mock_find_spec(name, package=None):
    if name == "peft" or name.startswith("peft."):
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = mock_find_spec

if "peft" in sys.modules:
    del sys.modules["peft"]

import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):

    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import argparse
import os
import json
from omegaconf import OmegaConf
from src.modules.image_encoder import ImageEncoder
from src.pipeline import Pipeline
from src.utils import set_ms_adapter
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPTextConfig, CLIPModel, CLIPTokenizer
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm

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

def load_custom_lora_weights(unet, lora_ckpt_path):

    lora_state_dict = load_file(lora_ckpt_path)

    for name, module in unet.named_modules():
        if hasattr(module, 'lora'):
            lora = module.lora

            key_a = f"unet.{name}.lora_A.weight"
            key_b = f"unet.{name}.lora_B.weight"
            if key_a in lora_state_dict and key_b in lora_state_dict:
                lora.lora_A.weight.data = lora_state_dict[key_a].to(lora.lora_A.weight.dtype)
                lora.lora_B.weight.data = lora_state_dict[key_b].to(lora.lora_B.weight.dtype)
            else:

                diffusers_name = name.replace(".", "_")
                key_a_diff = f"unet.{diffusers_name}.lora.down.weight"
                key_b_diff = f"unet.{diffusers_name}.lora.up.weight"
                if key_a_diff in lora_state_dict and key_b_diff in lora_state_dict:
                    lora.lora_A.weight.data = lora_state_dict[key_a_diff].to(lora.lora_A.weight.dtype)
                    lora.lora_B.weight.data = lora_state_dict[key_b_diff].to(lora.lora_B.weight.dtype)

    return unet

def load_lora_weights(ckpt_path):
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device=0) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict

def main():
    parser = argparse.ArgumentParser(description="Inference Script")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="../stabilityai/stable-diffusion-xl-base-1.0",
                        help="Pretrained model name or path")
    parser.add_argument("--clip_model_name_or_path", type=str,
                        default="../laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                        help="CLIP model name or path")
    parser.add_argument("--ms_ckpt", type=str,
                        default="../checkpoints/MS-Diffusion/ms_adapter.bin",
                        help="Path to MS adapter checkpoint")
    parser.add_argument("--clipself_ckpt", type=str,
                        default="./checkpoints/local_image_encoder/epoch_6.pt",
                        help="Path to CLIP self checkpoint (unused in GCN mode)")
    parser.add_argument("--gcn_ckpt", type=str,
                        default="../weight/final_gcn_weights.pt",
                        help="Path to trained multi-scale GCN weights (includes fine-grained fusion network)")
    parser.add_argument("--lora_ckpt", type=str,
                        default="../weight/pytorch_lora_weights.safetensors",
                        help="Path to trained UNet LoRA weights")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="LoRA rank, must match training configuration")
    parser.add_argument("--use_cross_attention", action="store_true", default=True,
                        help="Whether to use the noise-subject cross-attention module; must match the training configuration")
    parser.add_argument("--cross_attention_heads", type=int, default=8,
                        help="Number of cross-attention heads; must match the training configuration")
    parser.add_argument("--bidirectional_attention", action="store_true",
                        help="Whether to use bidirectional cross-attention; must match the training configuration")
    parser.add_argument("--no-bidirectional_attention", dest="bidirectional_attention", action="store_false",
                        help="Disable bidirectional cross-attention and use unidirectional mode; must match the training configuration")
    parser.add_argument("--enable_intra_type_attention", action="store_true", default=True,
                        help="Whether to enable same-type intra-attention (within noise and subject); must match the training configuration")
    parser.add_argument("--no-enable_intra_type_attention", dest="enable_intra_type_attention", action="store_false",
                        help="Disable same-type intra-attention; must match the training configuration")
    parser.add_argument("--intra_attention_type", type=str, default="local_window",
                        choices=["local_window", "axial", "global", "conv"],
                        help="Noise intra-attention type; must match the training configuration: local_window/axial/global/conv")
    parser.set_defaults(bidirectional_attention=True, enable_intra_type_attention=True)

    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--scale", type=float, default=0.6,
                        help="Scale parameter for MS adapter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--use_updated_noise_for_scheduler", action="store_true", default=False,
                        help="Experimental mode: use updated latent as the scheduler state for the next denoising step. Default keeps the current behavior.")

    parser.add_argument("--load_lora", action="store_true",
                        help="Whether to load LoRA weights")
    parser.add_argument("--no_load_lora", dest="load_lora", action="store_false",
                        help="Disable loading LoRA weights")
    parser.set_defaults(load_lora=True)

    parser.add_argument("--output_dir", type=str, default="../inference_result",
                        help="Output directory for generated images with multi-scale GCN")
    parser.add_argument("--negative_prompt", type=str,
                        default="monochrome, lowres, bad anatomy, worst quality, low quality",
                        help="Negative prompt for generation")

    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run the model on")

    parser.add_argument("--batch_mode", action="store_true",
                        help="Enable batch inference mode for dataset processing")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to the dataset directory for batch processing")
    parser.add_argument("--reference_image_dir", type=str, default="../data/evaluate_data/reference_images",
                        help="Directory containing reference images for dataset entities")
    parser.add_argument("--prompt_json_path", type=str, default="../data/evaluate_data/evaluate_prompt/prompt.json",
                        help="Path to the prompt.json file for custom batch processing")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from the specified index when doing batch inference (default: 0)")

    parser.add_argument("--evaluate", action="store_true",
                        help="Enable evaluation after generation")
    parser.add_argument("--real_image_dir", type=str, default=None,
                        help="Directory containing real images for FID calculation")
    parser.add_argument("--clip_model_name", type=str,
                        default="../laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                        help="CLIP model name for CLIP score calculation")

    args = parser.parse_args()

    import torch.distributed as dist
    def is_distributed():
        return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    rank = 0
    world_size = 1
    local_rank = 0

    if is_distributed():

        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'

        if rank != 0:
            import sys
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

    if not os.path.exists(args.reference_image_dir):
        print(f"⚠️  Warning: Reference image directory {args.reference_image_dir} does not exist")
        print("   Please verify the path or specify the correct path with --reference_image_dir")

    if not os.path.exists(args.gcn_ckpt):
        print(f"⚠️  Warning: GCN weight file {args.gcn_ckpt} does not exist")
        print("   Please verify the path or specify the correct path with --gcn_ckpt")

    if args.load_lora and not os.path.exists(args.lora_ckpt):
        print(f"⚠️  Warning: LoRA weight file {args.lora_ckpt} does not exist")
        print("   Please verify the path or specify the correct path with --lora_ckpt")
        print("   Add --no_load_lora to disable LoRA when it is not needed")

    if not os.path.exists(args.ms_ckpt):
        print(f"⚠️  Warning: MS adapter weight file {args.ms_ckpt} does not exist")
        print("   Please verify the path or specify the correct path with --ms_ckpt")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📂 Output will be saved to: {args.output_dir}")

    config = OmegaConf.load(args.config)

    ms_state_dict = torch.load(args.ms_ckpt)
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
        args.pretrained_model_name_or_path,
        subfolder="unet"
    ).to(args.device, torch.float16)
    adapter_modules, attn_store = set_ms_adapter(unet, scale=args.scale)
    adapter_modules.load_state_dict(ms_adapter_state_dict)

    if args.load_lora:

        unet_lora_parameters = add_lora_to_unet(
            unet,
            rank=args.lora_rank,
        )

        load_custom_lora_weights(unet, args.lora_ckpt)
        print(f"✅ UNet LoRA weights loaded successfully from {args.lora_ckpt}; rank={args.lora_rank}")
        print(f"✅ Number of LoRA parameters:{sum(p.numel() for p in unet_lora_parameters)/1e6:.2f}M")

    text_encoder_config = CLIPTextConfig.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )

    image_encoder = ImageEncoder(
        args.clip_model_name_or_path,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
        latent_init_mode="grounding",
        phrase_embeddings_dim=text_encoder_config.projection_dim,
        gcn_num_layers=2,
        use_cross_attention=args.use_cross_attention,
        cross_attention_heads=args.cross_attention_heads,
        bidirectional_attention=args.bidirectional_attention,
        enable_intra_type_attention=args.enable_intra_type_attention,
        intra_attention_type=args.intra_attention_type,
    ).to(args.device, dtype=torch.float16)

    image_encoder.load_state_dict(image_encoder_state_dict, strict=False)

    gcn_state_dict = torch.load(args.gcn_ckpt, map_location=args.device)
    for k, v in gcn_state_dict.items():
        gcn_state_dict[k] = v.to(torch.float16)

    has_fine_grained = any("fine_grained_fusion_net" in k for k in gcn_state_dict.keys())
    has_subject_fusion = any("subject_fusion_net" in k for k in gcn_state_dict.keys())
    has_cross_attention = any("cross_attention" in k for k in gcn_state_dict.keys())

    found_modules = []
    missing_modules = []
    if has_fine_grained:
        found_modules.append("fine_grained_fusion_net")
    else:
        missing_modules.append("fine_grained_fusion_net")

    if has_subject_fusion:
        found_modules.append("subject_fusion_net")
    else:
        missing_modules.append("subject_fusion_net")

    if has_cross_attention:
        found_modules.append("cross_attention")
    else:
        missing_modules.append("cross_attention")

    if found_modules:
        print(f"✅ GCN weights include modules:{', '.join(found_modules)}")
    if missing_modules:
        print(f"⚠️  Modules missing from GCN weights:{', '.join(missing_modules)}")

    image_encoder.load_state_dict(gcn_state_dict, strict=False)

    print(f"✅ GCN weights loaded successfully from {args.gcn_ckpt} and converted to float16 precision")

    pipe = Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        image_encoder=image_encoder,
    ).to(device=args.device, dtype=torch.float16)

    image_processor = CLIPImageProcessor()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch_mode:

        if args.prompt_json_path:

            with open(args.prompt_json_path, 'r') as f:
                prompt_data = json.load(f)

            if "combinations" in prompt_data:

                combinations = prompt_data["combinations"]
                prompts_per_combo = len(combinations[0]["prompts"]) if combinations else 0
                total_samples = len(combinations) * prompts_per_combo

                if is_distributed() and rank == 0:
                    print(f"🚀 Started distributed inference on {world_size} GPUs, with approximately {total_samples//world_size} samples")

                print(f"🚀 Starting combination-dataset batch inference with {len(combinations)}  combinations,{total_samples}  samples")

                def get_category_from_filename(filename):
                    filename = filename.lower()
                    if "person" in filename:
                        return "person"
                    elif filename in ["bear.jpg", "cat1.jpg", "cat2.jpg", "dog1.jpg", "dog2.jpg", "dog3.jpg",
                                     "fox.jpg", "hamster1.jpg", "hamster2.jpg", "owl1.jpg", "owl2.jpg",
                                     "parrot.jpg", "peacock.jpg", "rabbit.jpg", "sloth.jpg", "swan.jpg", "tiger.jpg"]:
                        return "animal"
                    elif "plushie" in filename:
                        return "plushie"
                    elif "toy" in filename:
                        return "toy"
                    elif "cartoon" in filename:
                        return "cartoon"
                    else:
                        return "object"

                existing_count = 0
                if args.start_idx == 0:

                    for f in os.listdir(args.output_dir):
                        if f.endswith('.jpg') and '_' in f:
                            try:
                                combo_idx = int(f.split('_')[0])
                                prompt_idx = int(f.split('_')[1].replace('.jpg', ''))

                                global_idx = combo_idx * prompts_per_combo + prompt_idx
                                if global_idx >= existing_count:
                                    existing_count = global_idx + 1
                            except:
                                pass
                    if existing_count > 0:
                        args.start_idx = existing_count
                        print(f"🔍 Detected {existing_count}  samples; resuming from index {args.start_idx} ")

                pbar = None
                progress_scanner = None
                if rank == 0:
                    pbar = tqdm(total=total_samples, initial=args.start_idx, desc="Inference progress", unit="sample")

                    import threading
                    import time
                    def scan_progress():
                        last_count = args.start_idx
                        while not pbar.disable:
                            time.sleep(10)
                            try:
                                current_count = len([f for f in os.listdir(args.output_dir) if f.endswith('.jpg')])
                                if current_count > last_count:
                                    pbar.update(current_count - last_count)
                                    last_count = current_count
                                if current_count >= total_samples:
                                    break
                            except:
                                pass
                    progress_scanner = threading.Thread(target=scan_progress, daemon=True)
                    progress_scanner.start()

                for combo_idx, combo in enumerate(combinations):

                    prompts_in_combo = len(combo["prompts"])

                    if (combo_idx + 1) * prompts_in_combo <= args.start_idx:

                        if rank == 0:
                            pbar.update(prompts_in_combo)
                        continue

                    subjects = combo["subjects"]
                    templates = combo["prompts"]
                    combination_type = combo["combination_type"]

                    subject_files = [subj["file"] for subj in subjects]
                    class_names = [subj["category"] for subj in subjects]
                    num_entities = len(subject_files)

                    input_images = []
                    valid_images = True
                    for image_path in subject_files:
                        if not os.path.exists(image_path):
                            print(f"⚠️  Combination {combo_idx} image not found {image_path}")
                            valid_images = False
                            break

                        img = Image.open(image_path).convert("RGB").resize((512, 512))
                        input_images.append(img)

                    if not valid_images or len(input_images) != num_entities:
                        print(f"⚠️  Combination {combo_idx} image loading incomplete; skipping")
                        continue

                    processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
                    processed_images = torch.stack(processed_images, dim=0).to(args.device, torch.float16)

                    default_boxes = []
                    for i in range(num_entities):

                        x1 = i / num_entities
                        x2 = (i + 1) / num_entities
                        default_boxes.append([x1, 0.1, x2, 0.9])
                    boxes = [default_boxes]

                    phrases = [class_names]

                    for prompt_idx, template in enumerate(templates):
                        global_idx = combo_idx * len(templates) + prompt_idx

                        if global_idx < args.start_idx:
                            if rank == 0:
                                pbar.update(1)
                            continue

                        if is_distributed() and global_idx % world_size != rank:
                            continue

                        output_path = os.path.join(args.output_dir, f"{combo_idx}_{prompt_idx}.jpg")
                        if os.path.exists(output_path):
                            print(f"⏭️  Combination {combo_idx} prompt {prompt_idx} already generated; skipping")
                            if rank == 0:
                                pbar.update(1)
                            continue

                        try:
                            prompt = template.format(*class_names)
                        except IndexError:

                            print(f"⚠️  Combination {combo_idx} template {template} placeholder count does not match the number of subjects; skipping")
                            if rank == 0:
                                pbar.update(1)
                            continue

                        generator = torch.Generator(unet.device).manual_seed(args.seed + global_idx)
                        with torch.no_grad():
                            images = pipe(
                                prompt=prompt,
                                negative_prompt=args.negative_prompt,
                                concept_images=processed_images,
                                num_inference_steps=args.num_inference_steps,
                                boxes=boxes,
                                phrases=phrases,
                                generator=generator,
                                num_images_per_prompt=args.num_samples,
                                use_updated_noise_for_scheduler=args.use_updated_noise_for_scheduler
                            ).images

                        if images:
                            img = images[0]
                            img.save(output_path)

                            metadata = {
                                "combination_id": combo["combination_id"],
                                "combination_type": combination_type,
                                "subject_files": subject_files,
                                "class_names": class_names,
                                "template": template,
                                "filled_prompt": prompt,
                                "prompt_index": prompt_idx,
                                "boxes": boxes,
                                "phrases": phrases
                            }
                            with open(os.path.join(args.output_dir, f"{combo_idx}_{prompt_idx}_metadata.json"), 'w') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)

                        pass

                if is_distributed():
                    dist.barrier()

                if rank == 0 and pbar is not None:

                    if progress_scanner is not None:
                        progress_scanner.join(timeout=5)

                    final_count = len([f for f in os.listdir(args.output_dir) if f.endswith('.jpg')])
                    pbar.n = final_count
                    pbar.refresh()
                    pbar.close()
                    print(f"\n🎉 Combination-dataset batch inference complete; all results saved to {args.output_dir}")
                    print(f"✅ Samples actually generated:{final_count}/{total_samples}")

                if is_distributed():
                    dist.destroy_process_group()

            else:

                prompts = prompt_data
                print(f"🚀 Starting batch inference with {len(prompts)}  samples")

                existing_count = 0
                if args.start_idx == 0:

                    for f in os.listdir(args.output_dir):
                        if f.endswith('.jpg') and f.replace('.jpg', '').isdigit():
                            existing_idx = int(f.replace('.jpg', ''))
                            if existing_idx >= existing_count:
                                existing_count = existing_idx + 1
                    if existing_count > 0:
                        args.start_idx = existing_count
                        print(f"🔍 Detected {existing_count}  samples; resuming from index {args.start_idx} ")

                for idx, prompt_data in enumerate(tqdm(prompts, desc="Generation progress")):

                    if idx < args.start_idx:
                        continue

                    output_path = os.path.join(args.output_dir, f"{idx}.jpg")
                    if os.path.exists(output_path):
                        print(f"⏭️  Index {idx} already generated; skipping")
                        continue

                    input_images = []
                    for image_path in prompt_data['images_path']:

                        full_image_path = None

                        candidate1 = os.path.join("..", image_path)

                        candidate2 = os.path.join(args.reference_image_dir, os.path.basename(image_path))

                        candidate3 = image_path

                        for candidate in [candidate1, candidate2, candidate3]:
                            if os.path.exists(candidate):
                                full_image_path = candidate
                                break

                        if not full_image_path:
                            print(f"⚠️  image not found {image_path}, attempted paths:")
                            print(f"   - {candidate1}")
                            print(f"   - {candidate2}")
                            print(f"   - {candidate3}")
                            print("   Skipping this sample")
                            continue

                        img = Image.open(full_image_path).convert("RGB").resize((512, 512))
                        input_images.append(img)

                    if len(input_images) != len(prompt_data['class_names']):
                        continue

                    processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
                    processed_images = torch.stack(processed_images, dim=0).to(args.device, torch.float16)

                    if 'boxes' in prompt_data:
                        boxes = prompt_data['boxes']
                    else:

                        num_entities = len(prompt_data['class_names'])
                        default_boxes = []
                        for i in range(num_entities):

                            x1 = i / num_entities
                            x2 = (i + 1) / num_entities
                            default_boxes.append([x1, 0.1, x2, 0.9])
                        boxes = [default_boxes]

                    phrases = prompt_data['phrases']

                    generator = torch.Generator(unet.device).manual_seed(args.seed + idx)
                    with torch.no_grad():
                        images = pipe(
                            prompt=prompt_data['prompt'],
                            negative_prompt=args.negative_prompt,
                            concept_images=processed_images,
                            num_inference_steps=args.num_inference_steps,
                            boxes=boxes,
                            phrases=phrases,
                            generator=generator,
                            num_images_per_prompt=args.num_samples,
                            use_updated_noise_for_scheduler=args.use_updated_noise_for_scheduler
                        ).images

                    if images:
                        img = images[0]
                        img.save(os.path.join(args.output_dir, f"{idx}.jpg"))

                        metadata = {
                            "prompt": prompt_data['prompt'],
                            "class_names": prompt_data['class_names'],
                            "images_path": prompt_data['images_path'],
                            "boxes": boxes,
                            "phrases": phrases
                        }
                        with open(os.path.join(args.output_dir, f"{idx}_metadata.json"), 'w') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"\n🎉 Batch inference complete; all results saved to {args.output_dir}")

        else:
            if not args.dataset_path:
                raise ValueError("dataset_path must be provided when using batch_mode without prompt_json_path")

            prompts_path = os.path.join(args.dataset_path, "prompts.json")
            with open(prompts_path, 'r') as f:
                dataset = json.load(f)

            print(f"🚀 Starting batch inference with {len(dataset)}  samples")

            for sample_idx, (sample_id, sample) in enumerate(dataset.items()):

                if sample_idx < args.start_idx:
                    continue
                print(f"\nProcessing sample {sample_id}/{len(dataset)-1}: {sample['prompt']}")

                sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_id}")
                os.makedirs(sample_output_dir, exist_ok=True)

                input_images = []
                for entity in sample['entities']:

                    entity_clean = entity.lower().replace(" ", "_")
                    image_path = None

                    for ext in ['.jpg', '.jpeg', '.png']:
                        candidate_path = os.path.join(args.reference_image_dir, f"{entity_clean}{ext}")
                        if os.path.exists(candidate_path):
                            image_path = candidate_path
                            break

                    if not image_path:
                        print(f"⚠️  No reference image found for entity {entity}; skipping this sample")
                        continue

                    img = Image.open(image_path).convert("RGB").resize((512, 512))
                    input_images.append(img)

                if len(input_images) != len(sample['entities']):
                    continue

                processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
                processed_images = torch.stack(processed_images, dim=0).to(args.device, torch.float16)

                num_entities = len(sample['entities'])
                default_boxes = []
                for i in range(num_entities):

                    x1 = i / num_entities
                    x2 = (i + 1) / num_entities
                    default_boxes.append([x1, 0.1, x2, 0.9])
                boxes = [default_boxes]

                phrases = [sample['entities']]

                generator = torch.Generator(unet.device).manual_seed(args.seed + int(sample_id))
                with torch.no_grad():
                    images = pipe(
                        prompt=sample['prompt'],
                        negative_prompt=args.negative_prompt,
                        concept_images=processed_images,
                        num_inference_steps=args.num_inference_steps,
                        boxes=boxes,
                        phrases=phrases,
                        generator=generator,
                        num_images_per_prompt=args.num_samples,
                        use_updated_noise_for_scheduler=args.use_updated_noise_for_scheduler
                    ).images

                for i, img in enumerate(images):
                    img.save(os.path.join(sample_output_dir, f"generated_{i}.jpg"))

                metadata = {
                    "prompt": sample['prompt'],
                    "entities": sample['entities'],
                    "predicate": sample['predicate'],
                    "boxes": boxes,
                    "phrases": phrases
                }
                with open(os.path.join(sample_output_dir, "metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"✅ Sample {sample_id} processed; results saved to {sample_output_dir}")

            print(f"\n🎉 Batch inference complete; all results saved to {args.output_dir}")

            generated_count = len([f for f in os.listdir(args.output_dir) if f.endswith('.jpg')])
            print(f"📊 Generated a total of {generated_count} images")

    else:

        input_images = []
        for image_path in config.images_path:

            full_image_path = None

            candidate1 = os.path.join("..", image_path)

            candidate2 = os.path.join(args.reference_image_dir, os.path.basename(image_path))

            candidate3 = image_path

            for candidate in [candidate1, candidate2, candidate3]:
                if os.path.exists(candidate):
                    full_image_path = candidate
                    break

            if not full_image_path:
                raise FileNotFoundError(f"image not found {image_path}, attempted paths:\n  - {candidate1}\n  - {candidate2}\n  - {candidate3}")

            img = Image.open(full_image_path).convert("RGB").resize((512, 512))
            input_images.append(img)

        processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
        processed_images = torch.stack(processed_images, dim=0)

        generator = torch.Generator(unet.device).manual_seed(args.seed)
        with torch.no_grad():
            image = pipe(
                prompt=config.prompt,
                negative_prompt=args.negative_prompt,
                concept_images=processed_images,
                num_inference_steps=args.num_inference_steps,
                boxes=config.boxes,
                phrases=config.phrases,
                generator=generator,
                num_images_per_prompt=args.num_samples,
                use_updated_noise_for_scheduler=args.use_updated_noise_for_scheduler
            ).images[0]
            image.save(os.path.join(args.output_dir, "output.jpg"))

        print(f"✅ Single-image inference complete; result saved to {os.path.join(args.output_dir, 'output.jpg')}")

    if args.device == "cuda":
        torch.cuda.empty_cache()
        print(f"\n🧹 GPU memory cleared")

if __name__ == "__main__":
    main()
