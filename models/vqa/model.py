"""
SatQuery-RS — VLM Model Loader (Singleton)
============================================
Loads GeoChat-7B base model with optional LoRA adapter.
Supports three inference modes:
  - quantized-4bit: 6GB VRAM (tight)
  - cpu-offload: 5GB VRAM + 20GB RAM (safe)
  - full-fp16: 16GB+ VRAM (cloud/demo)

This module is shared across VQA, Caption, and Grounding tools.

Usage:
    from models.vqa.model import get_model

    model, tokenizer, image_processor = get_model()
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ============================================================
# Environment-driven configuration
# ============================================================

def _get_config() -> dict[str, Any]:
    """Read inference configuration from environment."""
    from dotenv import load_dotenv
    load_dotenv()

    return {
        "base_model": os.getenv("GEOCHAT_BASE_MODEL", "MBZUAI/geochat-7B"),
        "lora_adapter": os.getenv("LORA_ADAPTER_PATH", "models/checkpoints/satquery-rs-vlm"),
        "vision_tower": os.getenv("VISION_TOWER", "openai/clip-vit-large-patch14-336"),
        "mode": os.getenv("SATQUERY_MODE", "cpu-offload"),
        "device_map": os.getenv("DEVICE_MAP", "auto"),
    }


# ============================================================
# Model loading
# ============================================================

class SatQueryVLM:
    """
    Singleton wrapper for the SatQuery-RS vision-language model.

    Handles:
    - Loading GeoChat-7B (LLaVA-1.5 architecture)
    - Applying LoRA adapter (if available)
    - Quantization and device placement
    - Shared access across VQA, caption, and grounding tools
    """

    _instance: SatQueryVLM | None = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.config = _get_config()
        self.device = None
        self.is_loaded = False
        self.model_name = "SatQuery-RS"

    @classmethod
    def get_instance(cls) -> SatQueryVLM:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Load the model, tokenizer, and image processor."""
        if self.is_loaded:
            return

        mode = self.config["mode"]
        base_model = self.config["base_model"]
        lora_adapter = self.config["lora_adapter"]

        logger.info(f"Loading SatQuery-RS VLM...")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  LoRA adapter: {lora_adapter}")

        try:
            self._load_model(mode, base_model, lora_adapter)
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully in '{mode}' mode")
            self._log_memory_usage()
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _load_model(self, mode: str, base_model: str, lora_adapter: str) -> None:
        """Load the base model and tokenizer with appropriate settings."""
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            CLIPImageProcessor,
        )
        from peft import PeftModel

        # Workaround for GeoChat config mapping issue
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.llama.configuration_llama import LlamaConfig
        if "geochat" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("geochat", LlamaConfig)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=False,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load image processor (CLIP ViT-L/14 @ 336px)
        logger.info("Loading image processor...")
        vision_tower = self.config["vision_tower"]
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        except Exception:
            # Fallback: try loading from base model
            self.image_processor = CLIPImageProcessor.from_pretrained(base_model)

        # Mode-specific model loading
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if mode == "quantized-4bit":
            logger.info("Loading with 4-bit NF4 quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        elif mode == "cpu-offload":
            logger.info("Loading with CPU offloading (safe mode for 6GB VRAM)...")
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["max_memory"] = {
                0: "5GiB",
                "cpu": "20GiB",
            }
            # Create offload directory
            offload_dir = Path("models/offload")
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_dir)

        elif mode == "full-fp16":
            logger.info("Loading in full FP16 (requires 16+ GB VRAM)...")
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        else:
            logger.warning(f"Unknown mode '{mode}', defaulting to cpu-offload")
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["max_memory"] = {0: "5GiB", "cpu": "20GiB"}

        # Load base model
        logger.info(f"Loading base model: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )

        # Apply LoRA adapter if available
        lora_path = Path(lora_adapter)
        if lora_path.exists() and (lora_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from: {lora_adapter}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_adapter,
                is_trainable=False,
            )
            self.model_name = "SatQuery-RS"
            logger.info("✅ LoRA adapter loaded — using SatQuery-RS adapted model")
        else:
            self.model_name = "GeoChat-7B (base)"
            logger.info(
                f"⚠️ No LoRA adapter found at {lora_adapter} — "
                f"using base GeoChat-7B (not adapted)"
            )

        self.model.eval()

        # Determine device
        if hasattr(self.model, "hf_device_map"):
            self.device = "auto"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def _log_memory_usage(self) -> None:
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info(
                f"GPU Memory: {allocated:.1f}GB allocated, "
                f"{reserved:.1f}GB reserved, "
                f"{total:.1f}GB total"
            )

    def generate(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = False,
        **kwargs,
    ) -> tuple[str, float]:
        """
        Generate text given input tokens and optional image tensor.

        Returns:
            (generated_text, confidence_score)
        """
        if not self.is_loaded:
            self.load()

        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            if images is not None:
                generate_kwargs["images"] = images

            generate_kwargs.update(kwargs)

            # Generate
            outputs = self.model.generate(
                input_ids,
                **generate_kwargs,
            )

            # Decode
            generated_ids = outputs.sequences[0]
            # Only decode the newly generated tokens
            new_tokens = generated_ids[input_ids.shape[-1]:]
            generated_text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            # Compute confidence from token scores
            confidence = self._compute_confidence(outputs)

        return generated_text, confidence

    def _compute_confidence(self, outputs) -> float:
        """
        Compute a confidence score from generation output.

        Uses mean token log-probability, normalized to [0, 1].
        """
        try:
            if hasattr(outputs, "scores") and outputs.scores:
                import torch.nn.functional as F

                log_probs = []
                for score in outputs.scores:
                    probs = F.softmax(score[0], dim=-1)
                    max_prob = probs.max().item()
                    log_probs.append(max_prob)

                if log_probs:
                    # Mean of max probabilities across generated tokens
                    confidence = sum(log_probs) / len(log_probs)
                    return round(min(max(confidence, 0.0), 1.0), 4)
        except Exception as e:
            logger.debug(f"Confidence computation failed: {e}")

        return 0.5  # Default confidence when computation fails

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None

        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")


def get_model() -> SatQueryVLM:
    """
    Get the loaded SatQueryVLM singleton.
    Loads the model on first call.
    """
    vlm = SatQueryVLM.get_instance()
    if not vlm.is_loaded:
        vlm.load()
    return vlm


# Allow running as a test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing model loading...")

    vlm = get_model()
    logger.info(f"Model: {vlm.model_name}")
    logger.info(f"Mode: {vlm.config['mode']}")
    logger.info(f"Device: {vlm.device}")
    logger.info("✅ Model loading test passed!")
