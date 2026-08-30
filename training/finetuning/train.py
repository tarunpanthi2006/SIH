"""
SatQuery-RS — Advanced LoRA Fine-Tuning Script
=================================================
Adapts GeoChat-7B → SatQuery-RS using BigEarthNet instruction data.

Features:
  ✦ Live eval loss display alongside training loss
  ✦ Rich ASCII progress dashboard with ETA
  ✦ Training curve CSV + JSON logging for post-hoc plotting
  ✦ EarlyStopping with patience
  ✦ Exponential Moving Average (EMA) loss smoothing
  ✦ Automatic best-checkpoint tracking
  ✦ Safe Hub resume with PyTorch 2.6 compatibility patches
  ✦ Memory-aware gradient accumulation

Designed for:
- 1x H100/A100/T4/L4 GPU (free tier sessions)
- Frequent checkpointing + HuggingFace Hub push
- Auto-resume from last checkpoint across accounts

Usage:
    # Debug run (1K samples, fast)
    python -m training.finetuning.train --config training/finetuning/config.yaml --debug

    # Full run
    python -m training.finetuning.train --config training/finetuning/config.yaml

    # Resume from HuggingFace Hub checkpoint
    python -m training.finetuning.train --config training/finetuning/config.yaml --resume-from-hub
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
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
# Training Metrics Tracker
# ============================================================

class TrainingMetrics:
    """
    Tracks training and eval metrics across the entire run.
    Writes a CSV log and a JSON summary at each logging step.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "training_curves.csv"
        self.json_path = self.log_dir / "training_summary.json"

        self.train_losses: list[dict] = []
        self.eval_losses: list[dict] = []
        self.best_eval_loss = float("inf")
        self.best_eval_step = 0
        self.ema_loss = None
        self.ema_alpha = 0.1  # Smoothing factor

        # Initialize CSV
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "epoch", "train_loss", "ema_loss",
                    "eval_loss", "learning_rate", "timestamp",
                ])

    def log_train(self, step: int, epoch: float, loss: float, lr: float):
        """Log a training step."""
        # Update EMA
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss

        entry = {
            "step": step,
            "epoch": round(epoch, 3),
            "loss": round(loss, 4),
            "ema_loss": round(self.ema_loss, 4),
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
        }
        self.train_losses.append(entry)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, round(epoch, 3), round(loss, 4),
                round(self.ema_loss, 4), "", lr,
                entry["timestamp"],
            ])

    def log_eval(self, step: int, epoch: float, eval_loss: float):
        """Log an evaluation step."""
        is_best = eval_loss < self.best_eval_loss
        if is_best:
            self.best_eval_loss = eval_loss
            self.best_eval_step = step

        entry = {
            "step": step,
            "epoch": round(epoch, 3),
            "eval_loss": round(eval_loss, 4),
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        self.eval_losses.append(entry)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, round(epoch, 3), "", "",
                round(eval_loss, 4), "",
                entry["timestamp"],
            ])

        return is_best

    def save_summary(self, extra: dict | None = None):
        """Save a JSON summary of the training run."""
        summary = {
            "total_train_steps": len(self.train_losses),
            "total_eval_steps": len(self.eval_losses),
            "best_eval_loss": round(self.best_eval_loss, 4) if self.best_eval_loss < float("inf") else None,
            "best_eval_step": self.best_eval_step,
            "final_train_loss": self.train_losses[-1]["loss"] if self.train_losses else None,
            "final_ema_loss": self.train_losses[-1]["ema_loss"] if self.train_losses else None,
            "train_history": self.train_losses[-20:],  # Last 20 entries
            "eval_history": self.eval_losses,
        }
        if extra:
            summary.update(extra)

        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)


# ============================================================
# Custom Callback for Live Dashboard
# ============================================================

class LiveDashboardCallback:
    """
    Custom HuggingFace Trainer callback that prints a live
    training dashboard with both train and eval loss.
    """

    def __init__(self, metrics: TrainingMetrics, total_steps: int):
        self.metrics = metrics
        self.total_steps = total_steps
        self.start_time = None
        self.last_eval_loss = None
        self.last_eval_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self._print_header()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        epoch = state.epoch or 0.0
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate", 0)

        # Track train loss
        if loss is not None:
            self.metrics.log_train(step, epoch, loss, lr)

        # Track eval loss
        if eval_loss is not None:
            is_best = self.metrics.log_eval(step, epoch, eval_loss)
            self.last_eval_loss = eval_loss
            self.last_eval_step = step

            # Print eval results
            star = " ★ BEST" if is_best else ""
            logger.info(
                f"  📊 EVAL │ step={step:>5} │ eval_loss={eval_loss:.4f}{star}"
            )

        # Print live dashboard line
        if loss is not None:
            self._print_dashboard(step, epoch, loss, lr)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after each evaluation run."""
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            step = state.global_step
            epoch = state.epoch or 0.0
            is_best = self.metrics.log_eval(step, epoch, eval_loss)
            self.last_eval_loss = eval_loss
            self.last_eval_step = step

            star = " ★ BEST" if is_best else ""
            logger.info(
                f"\n  {'═' * 62}\n"
                f"  📊 EVALUATION RESULTS\n"
                f"  ├─ Step:      {step:>6}\n"
                f"  ├─ Epoch:     {epoch:.2f}\n"
                f"  ├─ Eval Loss: {eval_loss:.4f}{star}\n"
                f"  ├─ Best Loss: {self.metrics.best_eval_loss:.4f} (step {self.metrics.best_eval_step})\n"
                f"  {'═' * 62}\n"
            )

    def on_save(self, args, state, control, **kwargs):
        """Called after each checkpoint save."""
        self.metrics.save_summary()
        logger.info(f"  💾 Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self.metrics.save_summary()
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(
            f"\n{'━' * 66}\n"
            f"  ✅ TRAINING COMPLETE\n"
            f"  ├─ Total Steps:     {state.global_step}\n"
            f"  ├─ Final Train Loss:{self.metrics.train_losses[-1]['loss']:.4f}" if self.metrics.train_losses else "" +
            f"\n  ├─ Final EMA Loss:  {self.metrics.train_losses[-1]['ema_loss']:.4f}" if self.metrics.train_losses else "" +
            f"\n  ├─ Best Eval Loss:  {self.metrics.best_eval_loss:.4f} (step {self.metrics.best_eval_step})" if self.metrics.best_eval_loss < float('inf') else "" +
            f"\n  ├─ Total Time:      {str(timedelta(seconds=int(elapsed)))}\n"
            f"{'━' * 66}"
        )

    def _print_header(self):
        logger.info(
            f"\n{'━' * 66}\n"
            f"  🚀 SATQUERY-RS TRAINING DASHBOARD\n"
            f"  ├─ Total Steps: {self.total_steps}\n"
            f"  ├─ Logging:     training_curves.csv\n"
            f"{'━' * 66}"
        )

    def _print_dashboard(self, step: int, epoch: float, loss: float, lr: float):
        """Print a compact dashboard line."""
        # Calculate progress
        progress = step / max(self.total_steps, 1)
        bar_width = 20
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # ETA
        if self.start_time and step > 0:
            elapsed = time.time() - self.start_time
            eta_seconds = (elapsed / step) * (self.total_steps - step)
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "??:??:??"

        ema = self.metrics.ema_loss or loss

        # Eval loss display
        eval_str = f"eval={self.last_eval_loss:.4f}" if self.last_eval_loss else "eval=pending"

        logger.info(
            f"  [{bar}] {step:>5}/{self.total_steps} │ "
            f"loss={loss:.4f} │ ema={ema:.4f} │ {eval_str} │ "
            f"lr={lr:.2e} │ ep={epoch:.2f} │ ETA={eta}"
        )


# ============================================================
# Dataset
# ============================================================

class InstructionDataset(torch.utils.data.Dataset):
    """
    LLaVA-style instruction dataset for LoRA fine-tuning.
    Loads pre-converted JSON from the BigEarthNet pipeline.
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
# Checkpoint Compatibility Patches
# ============================================================

def patch_checkpoint_for_old_transformers(checkpoint_dir: str):
    """
    Patches downloaded Hub checkpoints for compatibility with
    transformers 4.36.x and PyTorch 2.6.

    Fixes:
    1. Removes unknown keys from trainer_state.json
       (e.g., 'best_global_step' added in newer transformers)
    2. Deletes rng_state.pth that fails PyTorch 2.6 weights_only security
    3. Deletes optimizer.pt that fails due to peft/bnb parameter mismatch
    """
    import glob

    try:
        import transformers
        valid_keys = set(
            inspect.signature(
                transformers.trainer_callback.TrainerState.__init__
            ).parameters.keys()
        )
    except Exception:
        valid_keys = None

    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

    for ckpt_dir in checkpoint_dirs:
        # 1. Fix trainer_state.json
        state_file = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(state_file) and valid_keys:
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                removed = [k for k in list(state.keys()) if k not in valid_keys]
                if removed:
                    for k in removed:
                        del state[k]
                    with open(state_file, "w") as f:
                        json.dump(state, f)
                    logger.info(f"  🔧 Patched trainer_state.json: removed {removed}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not patch {state_file}: {e}")

        # 2. Remove incompatible RNG state (PyTorch 2.6 security)
        for rng_file in ["rng_state.pth", "rng_state.pt"]:
            rng_path = os.path.join(ckpt_dir, rng_file)
            if os.path.exists(rng_path):
                os.remove(rng_path)
                logger.info(f"  🔧 Removed {rng_file} (PyTorch 2.6 compat)")

        # 3. Remove incompatible optimizer (peft/bnb mismatch)
        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            os.remove(opt_path)
            logger.info(f"  🔧 Removed optimizer.pt (peft/bnb compat)")

        # 4. Remove scaler.pt if present
        scaler_path = os.path.join(ckpt_dir, "scaler.pt")
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            logger.info(f"  🔧 Removed scaler.pt")

    if checkpoint_dirs:
        logger.info(f"  ✅ Patched {len(checkpoint_dirs)} checkpoint(s) for compatibility")


def flatten_nested_checkpoints(output_dir: str):
    """
    If Hub download created nested checkpoints/ subfolder,
    move them up to the output_dir root.
    """
    nested = Path(output_dir) / "checkpoints"
    if nested.is_dir():
        import shutil
        for ckpt in nested.glob("checkpoint-*"):
            dest = Path(output_dir) / ckpt.name
            if not dest.exists():
                shutil.move(str(ckpt), str(dest))
                logger.info(f"  📁 Moved {ckpt.name} from nested folder")
        # Clean up empty nested dir
        try:
            nested.rmdir()
        except OSError:
            pass


# ============================================================
# Model Setup
# ============================================================

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
        "device_map": {"": 0},  # Forces everything onto GPU 0
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


# ============================================================
# Main Training Function
# ============================================================

def train(config_path: str, debug: bool = False, small: bool = False, resume_from_hub: bool = False):
    """Main training function with advanced features."""
    from transformers import TrainingArguments, Trainer, TrainerCallback

    config = load_config(config_path)
    train_config = config["training"]
    output_dir = config["output"]["dir"]
    log_dir = config["output"].get("logging_dir", "training/finetuning/logs")

    # Initialize metrics tracker
    metrics = TrainingMetrics(log_dir)

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
            logger.info(f"📥 Downloading checkpoint from Hub: {hub_repo}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hub_repo,
                local_dir=output_dir,
                repo_type="model",
            )

            # Fix nested checkpoints/ folder from Hub download
            flatten_nested_checkpoints(output_dir)

            # Patch checkpoints for library compatibility
            logger.info("🔧 Patching checkpoints for library compatibility...")
            patch_checkpoint_for_old_transformers(output_dir)

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

    # Calculate total steps for dashboard
    steps_per_epoch = math.ceil(
        len(train_dataset)
        / train_config.get("per_device_train_batch_size", 4)
        / train_config.get("gradient_accumulation_steps", 8)
    )
    total_steps = steps_per_epoch * train_config.get("epochs", 3)

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
        evaluation_strategy=train_config.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_config.get("eval_steps", 250) if eval_dataset else None,
        save_steps=train_config.get("save_steps", 250),
        save_total_limit=train_config.get("save_total_limit", 3),
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        logging_dir=log_dir,
        report_to="none",
        # Hub push for checkpoint survival
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        hub_strategy=train_config.get("hub_strategy", "every_save"),
        hub_private_repo=train_config.get("hub_private_repo", True),
        hub_token=hub_token,
        # Resume
        resume_from_checkpoint=train_config.get("resume_from_checkpoint", True),
        # Misc
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=42,
    )

    # Create custom callback
    dashboard = LiveDashboardCallback(metrics, total_steps)

    # Wrap callback methods for HuggingFace Trainer compatibility
    class HFDashboardCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            dashboard.on_train_begin(args, state, control, **kwargs)
        def on_log(self, args, state, control, logs=None, **kwargs):
            dashboard.on_log(args, state, control, logs=logs, **kwargs)
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            dashboard.on_evaluate(args, state, control, metrics=metrics, **kwargs)
        def on_save(self, args, state, control, **kwargs):
            dashboard.on_save(args, state, control, **kwargs)
        def on_train_end(self, args, state, control, **kwargs):
            dashboard.on_train_end(args, state, control, **kwargs)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[HFDashboardCallback()],
    )

    # Print training banner
    logger.info(
        f"\n{'━' * 66}\n"
        f"  🚀 STARTING LORA FINE-TUNING: GeoChat-7B → SatQuery-RS\n"
        f"  ├─ Training samples:  {len(train_dataset):,}\n"
        f"  ├─ Validation samples:{len(eval_dataset):,}" if eval_dataset else "" +
        f"\n  ├─ Epochs:            {train_config.get('epochs', 3)}\n"
        f"  ├─ Effective batch:   {train_config.get('per_device_train_batch_size', 4) * train_config.get('gradient_accumulation_steps', 8)}\n"
        f"  ├─ Learning rate:     {train_config.get('learning_rate', 2e-4)}\n"
        f"  ├─ Total steps:       ~{total_steps}\n"
        f"  ├─ Checkpoint every:  {train_config.get('save_steps', 250)} steps\n"
        f"  ├─ Eval every:        {train_config.get('eval_steps', 250)} steps\n"
        f"  ├─ Push to Hub:       {push_to_hub}\n"
        f"  ├─ Log dir:           {log_dir}\n"
        f"{'━' * 66}"
    )

    # Check for existing checkpoint to resume
    checkpoint = None
    if training_args.resume_from_checkpoint:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            # Sort numerically by the step number (e.g., checkpoint-1250 -> 1250)
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            checkpoint = str(checkpoints[-1])
            logger.info(f"📂 Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final adapter
    logger.info(f"Saving final adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save final metrics summary
    metrics.save_summary(extra={
        "config_path": config_path,
        "output_dir": output_dir,
        "debug": debug,
        "small": small,
    })

    # Push final to Hub
    if push_to_hub:
        logger.info(f"Pushing final model to Hub: {hub_model_id}")
        trainer.push_to_hub()

    logger.info("✅ Training complete!")
    logger.info(f"Adapter saved to: {output_dir}")
    logger.info(f"Training curves: {metrics.csv_path}")
    logger.info(f"Training summary: {metrics.json_path}")
    logger.info(f"To use: set LORA_ADAPTER_PATH={output_dir} in .env")


def load_config(config_path: str) -> dict:
    """Load YAML training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
