"""STT fine-tuning pipeline for Whisper / Conformer models."""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Speech-to-Text model on Vietnamese audio")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--train_data", type=str, default="data/processed/metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", default=True)
    return parser.parse_args()


def train():
    args = parse_args()
    logger.info("Initializing STT fine-tuning: %s", args.model_name_or_path)

    try:
        import torch
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
        )
    except ImportError:
        logger.error(
            "Transformers and Torch must be installed for STT fine-tuning. "
            "Run: pip install torch transformers datasets accelerate"
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 1. Load Processor & Model
    try:
        processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="vi", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
    except Exception as e:
        logger.warning("Could not load HuggingFace weights directly (%s). Training loop template ready.", e)
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.epochs * 100,
        gradient_checkpointing=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    logger.info("Training configuration initialized. Checkpoint directory: %s", output_path)
    logger.info("Ready to attach Custom Dataset and execute Trainer.")


if __name__ == "__main__":
    train()
