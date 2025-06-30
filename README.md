# PAO: Project Adapt Overcome

## Overview

PAO (Project Adapt Overcome) is a framework for training and evaluating multimodal models, with a focus on vision-language capabilities. The project supports various models including InternVL, InternLLM2, Phi3, and others.

## Project Structure

- **models/**: Contains model implementations and configurations
  - `InternLLM2/`: InternLLM2 model implementation
  - `InternVIT/`: InternVIT model implementation
  - `Phi3/`: Phi3 model implementation
  - `multimodal/`: Multimodal model implementations
  - `conversation.py`: Conversation handling utilities

- **train/**: Training utilities and scripts
  - `pretrain.py`: Main pretraining script
  - `dataset.py`: Dataset loading and preprocessing
  - `patch.py`: Model patching utilities
  - `constants.py`: Constants used in training

- **eval/**: Evaluation scripts and utilities
  - `caption/`: Image captioning evaluation

## Getting Started

To get started with PAO, clone this repository and explore the training and evaluation scripts in the respective directories.

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- CUDA-compatible GPU (recommended)

### Training a Model

You can train a multimodal model using the `pretrain.py` script:

```bash
python train/pretrain.py \
  --checkpoint /path/to/base/model \
  --output_dir /path/to/output \
  --train_data_path /path/to/training/data \
  --do_train \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5
```

### Evaluating Image Captioning

To evaluate a model on image captioning tasks:

```bash
python eval/caption/evaluate_caption.py \
  --checkpoint /path/to/model/checkpoint \
  --datasets coco,flickr30k \
  --batch-size 1 \
  --num-beams 5 \
  --out-dir results
```

## Supported Models

PAO currently supports the following models:
- InternVL
- InternLLM2
- Phi3
- And other vision-language models

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.
