# Knowledge Distillation with KL Divergence

This repository implements knowledge distillation using KL divergence, a technique where a lightweight "student" model is trained to mimic the behavior of a larger, more complex "teacher" model. The core idea is to transfer knowledge from the teacher to the student by minimizing the KL divergence between their output distributions, alongside the standard classification loss on true labels.


## Overview
Knowledge distillation is a model compression method that leverages the "knowledge" captured by a high-performance teacher model to train a smaller student model. This implementation focuses on transferring knowledge via KL divergence (a measure of difference between probability distributions) to ensure the student not only learns from true labels but also mimics the teacher's output patterns, often leading to better generalization than training the student alone.


## Key Features
- **Dataset Handling**: Loading and preprocessing the CIFAR-10 dataset for training and evaluation.
- **Model Definitions**: 
  - A complex, high-capacity teacher model.
  - A lightweight student model designed for efficiency.
- **Training Functions**:
  - Baseline training (using standard cross-entropy loss) for both teacher and student models.
  - Distillation training that combines:
    - KL divergence loss (to align student outputs with teacher outputs).
    - Cross-entropy loss (to align student outputs with true labels).
- **Evaluation**: Tools to compare performance (e.g., accuracy) across the teacher, baseline student, and distillation-trained student.


## Usage
1. Prepare dependencies (e.g., PyTorch, torchvision) for CIFAR-10 handling and model training.
2. Run training scripts to:
   - Train the teacher model using standard cross-entropy loss.
   - Train the baseline student model (without distillation) using cross-entropy loss.
   - Train the student with knowledge distillation, combining KL divergence (against teacher outputs) and cross-entropy loss (against true labels).
3. Use evaluation code to compare performance metrics (e.g., test accuracy) of all models.


This implementation demonstrates how knowledge distillation can help a lightweight student model approximate the performance of a larger teacher model.