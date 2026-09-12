# DRG-Diff: Dual Relation Graph Awareness-Driven Multi-Subject Personalized Image Generation

## Introduction

DRG-Diff is a relation-aware multi-subject image customization framework. Given multiple reference subjects and a text prompt describing their interaction, the model aims to preserve the identity of every subject while generating accurate and natural spatial or semantic relations.

The framework builds on multi-subject diffusion and introduces graph-based subject modeling, joint noise-subject reasoning, and cross-attention constraint mechanisms to improve subject consistency and relation generation.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Model Preparation

Download the pretrained MS-Diffusion model:

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p checkpoints
huggingface-cli download doge1516/MS-Diffusion --local-dir ./checkpoints/MS-Diffusion
```

Update the model and checkpoint paths in the configuration files before training or inference.

## Data Preparation

Organize the training and evaluation data as follows:

```text
data/
├── evaluate_data/
│   ├── evaluate_prompt/
│   └── reference_images/
└── train_data/
    ├── objects/
    └── samples/
        ├── sample_000001/
        │   ├── gt.jpg
        │   ├── metadata.json
        │   ├── ref_1.jpg
        │   └── ref_2.jpg
        ├── sample_000002/
        ├── sample_000003/
        └── ...
```

Each directory under `data/train_data/samples/` represents one training sample and contains:

- `gt.jpg`: the ground-truth image containing the subjects and their relation.
- `metadata.json`: the prompt, subject information, bounding boxes, and other sample metadata.
- `ref_1.jpg`, `ref_2.jpg`, ...: reference images for the corresponding subjects.

The `data/train_data/objects/` directory stores the source object images. Evaluation prompts and their reference images are stored under `data/evaluate_data/evaluate_prompt/` and `data/evaluate_data/reference_images/`, respectively.

Image extensions may be changed to match the files used by your dataset loader.

## Training

Configure Accelerate before launching training. Multi-GPU training is recommended.

```bash
accelerate config
```

Replace the model, dataset, and output paths in the training configuration, then run:

```bash
mkdir -p lora-weights
bash train.sh
```

## Inference

Before inference, set the checkpoint, dataset, and output paths in `configs/batch_default.yaml`.

### Single-process inference

```bash
python inference.py \
  --config configs/batch_default.yaml \
  --batch_mode \
  --use_cross_attention \
  --cross_attention_heads=8 \
  --intra_attention_type="conv" \
  --num_inference_steps 50
```

### Distributed inference

```bash
bash inference_distributed.sh
```

## Evaluation

After generation, configure the generated-image and reference-data paths required by the evaluation script, then run:

```bash
python Multi_evaluate_generation.py
```

## Acknowledgements

This project is built upon the excellent work of [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion) and [DreamRelation](https://github.com/Shi-qingyu/DreamRelation). We sincerely thank the authors for releasing their code and models to the research community.

## Citation

If you find this repository useful for your research, please cite the DRG-Diff paper. The official BibTeX entry will be added upon publication.
