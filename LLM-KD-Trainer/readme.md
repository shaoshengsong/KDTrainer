# Large Language Model Knowledge Distillation with LoRA

This repository implements knowledge distillation for large language models (LLMs) using the Hugging Face Transformers and Peft libraries. It enables efficient knowledge transfer from a pre-trained teacher LLM to a smaller student model through parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation), combined with KL divergence loss and cross-entropy loss for stable knowledge migration.

## Overview

Knowledge distillation for large language models aims to compress the capabilities of a large, high-performance teacher model into a more lightweight student model while retaining key functionalities. This implementation leverages:



*   Hugging Face Transformers for model loading and training pipelines

*   Peft library's LoRA technology for parameter-efficient student model fine-tuning (reducing computational overhead)

*   Hybrid loss function (KL divergence + cross-entropy) to align student outputs with both teacher distributions and ground truth labels

## Key Features



*   **LLM-focused Pipeline**: Optimized for large language model distillation, supporting common pre-trained model architectures.

*   **Parameter-Efficient Fine-Tuning**: Uses LoRA to adapt student model parameters, significantly reducing memory and computation requirements compared to full fine-tuning.

*   **Dual Loss Mechanism**:


    *   KL divergence loss to mimic teacher model's output distributions

    *   Cross-entropy loss to ensure alignment with ground truth labels

*   **Compatibility**: Seamlessly integrates with Hugging Face ecosystem (Transformers, Datasets, Peft) for streamlined workflow.

Python: 3.12.9
PyTorch: 2.6.0+cu124
Transformers: 4.55.0  (20250809)
PEFT: 0.15.2
PyYAML: 6.0.2