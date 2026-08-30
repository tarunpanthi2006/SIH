"""
SatQuery-RS — LoRA Fine-Tuning Script
=======================================
Adapts GeoChat-7B → SatQuery-RS using BigEarthNet.txt instruction data.

Designed for:
- 1x H100 80GB (3-hour free sessions, multiple accounts)
- Frequent checkpointing + HuggingFace Hub push for checkpoint survival
- Auto-resume from last checkpoint when switching accounts

Usage:
    # Debug run (1K samples, fast)
    python -m training.finetuning.train --config training/finetuning/config.yaml --debug

    # Small run (10K samples)
    python -m training.finetuning.train --config training/finetuning/config.yaml --small

    # Full run
    python -m training.finetuning.train --config training/finetuning/config.yaml

    # Resume from HuggingFace Hub checkpoint
    python -m training.finetuning.train --config training/finetuning/config.yaml --resume-from-hub
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Dataset
# ============================================================

class InstructionDataset(torch.utils.data.Dataset):
    """
    LLaVA-style instruction dataset for LoRA fine-tuning.
    Loads pre-converted JSON from the BigEarthNet.txt pipeline.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        max_length: int = 2048,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        logger.info(f"Loaded {len(self.data)} instruction samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # Build conversation text
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            # Skip malformed samples
            return self.__getitem__((idx + 1) % len(self.data))

        human_turn = conversations[0].get("value", "")
        gpt_turn = conversations[1].get("value", "")

        # Format as: USER: <image>\n{question}\nASSISTANT: {answer}</s>
        prompt = f"USER: {human_turn}\nASSISTANT: {gpt_turn}</s>"

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels (mask the prompt, only compute loss on assistant's response)
        labels = input_ids.clone()
        # Find where ASSISTANT: starts and mask everything before
        assistant_token = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        prompt_len = self._find_subsequence(input_ids.tolist(), assistant_token)
        if prompt_len > 0:
            labels[:prompt_len] = -100  # Ignore prompt in loss

        # Load image if available
        image_path = sample.get("image", "")
        pixel_values = None
        if image_path and os.path.exists(image_path):
            try:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                processed = self.image_processor.preprocess(image, return_tensors="pt")
                pixel_values = processed["pixel_values"].squeeze(0)
            except Exception as e:
                logger.debug(f"Failed to load image {image_path}: {e}")

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result

    @staticmethod
    def _find_subsequence(sequence: list, subsequence: list) -> int:
        """Find the end position of a subsequence in a sequence."""
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return i + len(subsequence)
        return 0


# ============================================================
# Training
# ============================================================

def load_config(config_path: str) -> dict:
    """Load YAML training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """Load base model with quantization and apply LoRA."""
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        CLIPImageProcessor,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base_model = config["model"]["base"]
    lora_config = config["lora"]
    quant_config = config.get("quantization", {})

    # Workaround for GeoChat config mapping issue
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.llama.configuration_llama import LlamaConfig
    if "geochat" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("geochat", LlamaConfig)

    # Tokenizer
    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Image processor
    vision_tower = config["model"].get("vision_tower", "openai/clip-vit-large-patch14-336")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    # Quantization config for QLoRA
    bnb_config = None
    if quant_config.get("load_in_4bit", False):
        logger.info("Setting up 4-bit QLoRA quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )

    # Load base model
    logger.info(f"Loading base model: {base_model}")
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Prepare for k-bit training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Apply LoRA
    logger.info("Applying LoRA adapter")
    peft_config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, tokenizer, image_processor


def train(config_path: str, debug: bool = False, small: bool = False, resume_from_hub: bool = False):
    """Main training function."""
    from transformers import TrainingArguments, Trainer

    config = load_config(config_path)
    train_config = config["training"]
    output_dir = config["output"]["dir"]

    # Override data path for debug/small modes
    if debug:
        data_train = config["data"]["train"].replace("train.json", "debug.json")
        train_config["epochs"] = 1
        train_config["save_steps"] = 50
        train_config["eval_steps"] = 50
        logger.info("🔧 DEBUG MODE: Using debug.json, 1 epoch")
    elif small:
        data_train = config["data"]["train"].replace("train.json", "train_small.json")
        logger.info("🔧 SMALL MODE: Using train_small.json")
    else:
        data_train = config["data"]["train"]

    data_val = config["data"].get("val", None)

    # Download checkpoint from Hub if resuming
    if resume_from_hub:
        hub_repo = os.getenv("HF_HUB_REPO", config["training"].get("hub_model_id", ""))
        if hub_repo:
            logger.info(f"Downloading checkpoint from Hub: {hub_repo}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hub_repo,
                local_dir=output_dir,
                repo_type="model",
            )

    # Setup model
    model, tokenizer, image_processor = setup_model_and_tokenizer(config)

    # Setup datasets
    logger.info(f"Loading training data: {data_train}")
    train_dataset = InstructionDataset(
        data_train, tokenizer, image_processor,
        max_length=train_config.get("max_length", 2048),
    )

    eval_dataset = None
    if data_val and os.path.exists(data_val):
        logger.info(f"Loading validation data: {data_val}")
        eval_dataset = InstructionDataset(
            data_val, tokenizer, image_processor,
            max_length=train_config.get("max_length", 2048),
        )

    # Hub configuration
    hub_model_id = os.getenv("HF_HUB_REPO", config["training"].get("hub_model_id", ""))
    push_to_hub = bool(hub_model_id) and train_config.get("push_to_hub", True)
    hub_token = os.getenv("HF_TOKEN", None)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config.get("epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        learning_rate=train_config.get("learning_rate", 2e-4),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        weight_decay=train_config.get("weight_decay", 0.0),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        bf16=train_config.get("bf16", True),
        fp16=train_config.get("fp16", False),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        logging_steps=train_config.get("logging_steps", 10),
        eval_strategy=train_config.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_config.get("eval_steps", 250) if eval_dataset else None,
        save_steps=train_config.get("save_steps", 250),
        save_total_limit=train_config.get("save_total_limit", 3),
        logging_dir=config["output"].get("logging_dir", "training/finetuning/logs"),
        report_to="none",  # Set to "wandb" if wandb is enabled
        # Hub push for checkpoint survival
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        hub_strategy=train_config.get("hub_strategy", "every_save") if push_to_hub else None,
        hub_private_repo=train_config.get("hub_private_repo", True),
        hub_token=hub_token,
        # Resume
        resume_from_checkpoint=train_config.get("resume_from_checkpoint", True),
        # Misc
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=42,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting LoRA fine-tuning: GeoChat-7B → SatQuery-RS")
    logger.info(f"  Training samples: {len(train_dataset):,}")
    if eval_dataset:
        logger.info(f"  Validation samples: {len(eval_dataset):,}")
    logger.info(f"  Epochs: {train_config.get('epochs', 3)}")
    logger.info(f"  Effective batch size: {train_config.get('per_device_train_batch_size', 4) * train_config.get('gradient_accumulation_steps', 8)}")
    logger.info(f"  Checkpoint save every: {train_config.get('save_steps', 250)} steps")
    logger.info(f"  Push to Hub: {push_to_hub}")
    logger.info("=" * 60)

    # Check for existing checkpoint to resume
    checkpoint = None
    if training_args.resume_from_checkpoint:
        checkpoints = sorted(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoint = str(checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final adapter
    logger.info(f"Saving final adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push final to Hub
    if push_to_hub:
        logger.info(f"Pushing final model to Hub: {hub_model_id}")
        trainer.push_to_hub()

    logger.info("✅ Training complete!")
    logger.info(f"Adapter saved to: {output_dir}")
    logger.info(f"To use: set LORA_ADAPTER_PATH={output_dir} in .env")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune GeoChat-7B → SatQuery-RS")
    parser.add_argument(
        "--config",
        type=str,
        default="training/finetuning/config.yaml",
        help="Path to training config",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode: 1K samples, 1 epoch")
    parser.add_argument("--small", action="store_true", help="Small mode: 10K samples")
    parser.add_argument("--resume-from-hub", action="store_true", help="Download & resume from HF Hub")

    args = parser.parse_args()
    train(args.config, debug=args.debug, small=args.small, resume_from_hub=args.resume_from_hub)


if __name__ == "__main__":
    main()
