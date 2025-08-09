
"""
Large Language Model Knowledge Distillation Trainer
Author: shaoshengsong
Created: 2025-05-20
Description:
    Implements large language model knowledge distillation based on the Hugging Face Transformers and Peft libraries.
    Performs parameter-efficient fine-tuning on the student model using LoRA (Low-Rank Adaptation) technology.
    Combines KL divergence loss and cross-entropy loss to transfer knowledge from the teacher model to the student model.
"""

import yaml
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# Configure logging: Set format and level (INFO to capture all key steps)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ConfigHandler:
    """Handles loading and management of configuration parameters from YAML files."""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """Load configuration parameters from a YAML file."""
        logger.info(f"Loading configuration from file: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration with keys: {list(config.keys())}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse configuration: {str(e)}")
            raise


class TextDataset(Dataset):
    """Dataset class for causal language modeling tasks with instruction-input-output format."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizerBase, 
        max_sequence_length: int
    ) -> None:
        logger.info(f"Initializing TextDataset with data path: {data_path}")
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        logger.info(f"Using padding token ID: {self.pad_token_id} (EOS token as fallback if pad token undefined)")
        
        self.data = self._load_and_validate_data()
        logger.info(f"Dataset initialized with {len(self.data)} valid samples")

    def _load_and_validate_data(self) -> List[Dict[str, str]]:
        """Load and validate JSON dataset structure."""
        logger.info(f"Loading and validating data from: {self.data_path}")
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise RuntimeError(f"Top-level data must be a list, got {type(data)}")
            logger.debug(f"Loaded raw data with {len(data)} entries")

            required_keys = {"instruction", "input", "output"}
            valid_samples = []
            for idx, sample in enumerate(data):
                if isinstance(sample, dict) and required_keys.issubset(sample.keys()):
                    valid_samples.append(sample)
                else:
                    missing_keys = required_keys - sample.keys() if isinstance(sample, dict) else required_keys
                    logger.warning(f"Skipping invalid sample at index {idx}: Missing keys {missing_keys}")
            
            logger.info(f"Validated {len(valid_samples)}/{len(data)} samples (filtered invalid entries)")
            return valid_samples
        except Exception as e:
            logger.error(f"Data loading/validation failed: {str(e)}")
            raise

    def _preprocess_sample(self, sample: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Preprocess a single sample into model-ready tensors."""
        # (Existing preprocessing logic remains unchanged)
        query = "".join([sample["instruction"], sample["input"]])
        answer = f"{sample['output']}{self.tokenizer.eos_token}"
        
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)
        
        input_ids = input_ids[:self.max_sequence_length]
        labels = labels[:self.max_sequence_length]
        attention_mask = attention_mask[:self.max_sequence_length]
        
        pad_length = self.max_sequence_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_length
        labels += [-100] * pad_length
        attention_mask += [0] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._preprocess_sample(self.data[index])


class KnowledgeDistillationTrainer(Trainer):
    """Custom Trainer for knowledge distillation with teacher-student framework."""
    
    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        distillation_config: Dict[str, Any],
        use_entropy_loss: bool = False,** kwargs
    ) -> None:
        logger.info("Initializing KnowledgeDistillationTrainer")
        super().__init__(model=student_model, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.distillation_config = distillation_config
        self.use_entropy_loss = use_entropy_loss
        logger.info(f"Distillation config: Temperature={distillation_config['temperature']}, Padding ID={distillation_config['padding_id']}")
        logger.info(f"Loss strategy: {'Combined KL + Cross-Entropy' if use_entropy_loss else 'Pure KL Divergence'}")

    @staticmethod
    def compute_forward_kl_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_labels: torch.Tensor,
        padding_id: int,
        reduction: str = "sum",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # (Existing logic remains unchanged)
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        student_log_probs = torch.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, dim=-1, dtype=torch.float32)

        kl_divergence = teacher_probs * (teacher_log_probs - student_log_probs)
        kl_divergence = kl_divergence.sum(dim=-1)

        if reduction == "sum":
            pad_mask = target_labels.eq(padding_id)
            kl_divergence = kl_divergence.masked_fill_(pad_mask, 0.0)
            kl_divergence = kl_divergence.sum()

        return kl_divergence

    def compute_loss(
        self, 
        model: PreTrainedModel, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None 
    ) -> Tuple[torch.Tensor, Any] | torch.Tensor:
        # Forward pass with student
        student_outputs = model(** inputs)
        student_loss = student_outputs.loss
        student_logits = student_outputs.logits

        if student_loss.dim() > 0:  
            student_loss = student_loss.mean()  

        # Forward pass with teacher (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits


        # Align vocabulary sizes if mismatch
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[:, :, :min_vocab]
            teacher_logits = teacher_logits[:, :, :min_vocab]
            logger.debug(f"Aligned vocabulary sizes to {min_vocab} (smaller of student/teacher)")

        # Compute KL loss
        kl_loss = self.compute_forward_kl_divergence(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            target_labels=inputs["labels"],
            padding_id=self.distillation_config["padding_id"],
            temperature=self.distillation_config["temperature"]
        )

        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()
        student_loss = student_loss.unsqueeze(0)
        kl_loss = kl_loss.unsqueeze(0)

        # Combine losses
        total_loss = 0.7 * kl_loss + 0.3 * student_loss if self.use_entropy_loss else kl_loss
        total_loss = total_loss.unsqueeze(0)

        # Log loss components (every 10 steps to avoid spam)
        if self.state.global_step % 10 == 0:
            logger.info(
                f"Step {self.state.global_step} - "
                f"KL Loss: {kl_loss.item():.4f} | "
                f"Student CE Loss: {student_loss.item():.4f} | "
                f"Total Loss: {total_loss.item():.4f}"
            )

        return (total_loss, student_outputs) if return_outputs else total_loss


class ModelLoader:
    """Utility for loading models, tokenizers, and applying LoRA."""
    
    @staticmethod
    def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
        logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning(f"Tokenizer has no pad token; using EOS token ('{tokenizer.eos_token}') as fallback")
        logger.info(f"Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")
        return tokenizer

    @staticmethod
    def load_student_model(
        model_path: str, 
        lora_config: LoraConfig,
        device: str = "cuda"
    ) -> PreTrainedModel:
        logger.info(f"Loading student model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        logger.info(f"Applying LoRA configuration to student model (rank={lora_config.r})")
        model = get_peft_model(model, lora_config)
        model = model.to(device)
        logger.info(f"Student model moved to device: {device}")
        return model

    @staticmethod
    def load_teacher_model(
        model_path: str, 
        lora_path: Optional[str] = None,
        device: str = "cuda"
    ) -> PreTrainedModel:
        logger.info(f"Loading teacher model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if lora_path:
            logger.info(f"Loading LoRA adapters for teacher from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            
        model = model.to(device)
        model.eval()
        logger.info(f"Teacher model moved to device: {device} (set to evaluation mode)")
        return model


def main() -> None:
    """Main pipeline for knowledge distillation training."""
    logger.info("===== Starting Large Language Model Knowledge Distillation =====")

    # 1. Load configuration
    config = ConfigHandler.load_config("config.yaml")

    # 2. Load tokenizer
    tokenizer = ModelLoader.load_tokenizer(config["student_path"])

    # 3. Configure LoRA for student model
    logger.info("Configuring LoRA for parameter-efficient fine-tuning")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target"].split(","),
        lora_dropout=config["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(f"LoRA target modules: {lora_config.target_modules}")

    # 4. Load student model with LoRA
    student_model = ModelLoader.load_student_model(
        model_path=config["student_path"],
        lora_config=lora_config,
        device="cuda"
    )
    logger.info("Student model trainable parameters:")
    student_model.print_trainable_parameters()  # Built-in method for parameter stats

    # 5. Load teacher model
    teacher_model = ModelLoader.load_teacher_model(
        model_path=config["teacher_path"],
        lora_path=config.get("teacher_lora_path"),
        device="cuda"
    )

    # 6. Prepare dataset
    logger.info(f"Preparing training dataset from: {config['data_path']}")
    dataset = TextDataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        max_sequence_length=config["max_sequence_length"]
    )
    logger.info(f"Dataset prepared - Total samples: {len(dataset)}")
    data_collator = DefaultDataCollator()
    logger.info("Using DefaultDataCollator for batching")

    # 7. Configure training arguments
    logger.info("Configuring training arguments")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        do_train=True,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=config["logging_steps"],
        report_to=config["report_to"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
        bf16=config["bf16"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_pin_memory=config["dataloader_pin_memory"],
    )
    logger.info(f"Training arguments set - Epochs: {config['num_train_epochs']}, Batch size: {config['per_device_train_batch_size']}")

    # 8. Initialize optimizer and scheduler
    logger.info(f"Initializing optimizer (AdamW) with learning rate: {config['learning_rate']}")
    optimizer = optim.AdamW(student_model.parameters(), lr=config["learning_rate"])
    
    total_training_steps = (
        len(dataset)
        // (config["per_device_train_batch_size"] * config["gradient_accumulation_steps"])
        * config["num_train_epochs"]
    )
    logger.info(f"Total training steps calculated: {total_training_steps}")
    
    lr_scheduler = get_scheduler(
        name=config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_training_steps,
    )
    logger.info(f"Learning rate scheduler initialized: {config['lr_scheduler_type']} (warmup steps: {config['warmup_steps']})")

    # 9. Configure distillation parameters
    distillation_config = {
        "temperature": config["temperature"],
        "padding_id": config["padding_id"]
    }

    # 10. Initialize trainer
    logger.info("Initializing custom KnowledgeDistillationTrainer")
    trainer = KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_config=distillation_config,
        use_entropy_loss=config["use_entropy_loss"],
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
    )

    # 11. Start training
    logger.info("===== Starting training process =====")
    logger.info(f"Training output will be saved to: {config['output_dir']}")
    trainer.train(resume_from_checkpoint=False)

    # 12. Save final model
    logger.info("Training completed. Saving final model and state...")
    trainer.save_model(config["save_model_dir"])
    trainer.save_state()
    logger.info(f"Final model saved to: {config['save_model_dir']}")
    logger.info("===== Knowledge distillation pipeline completed successfully =====")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
