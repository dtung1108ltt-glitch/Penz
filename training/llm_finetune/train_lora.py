"""
Script huấn luyện QLoRA Fine-Tuning cho mô hình ngôn ngữ (LLM) phục vụ Voice AI.
Nạp tri thức Bách khoa Toàn thư Việt Nam (vietnam_knowledge_full.jsonl) vào Qwen2.5-3B-Instruct.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Đảm bảo UTF-8 terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QLoRA_Trainer")


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning cho Voice AI trên tri thức Bách khoa toàn thư Việt Nam")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face Model ID hoặc đường dẫn cục bộ")
    parser.add_argument("--data_file", type=str, default="data/processed/vietnam_knowledge_full.jsonl", help="Đường dẫn file ChatML dataset")
    parser.add_argument("--output_dir", type=str, default="training/llm_finetune/checkpoints", help="Thư mục lưu checkpoints")
    parser.add_argument("--adapter_dir", type=str, default="training/llm_finetune/checkpoints/vietnam_voice_adapter", help="Thư mục xuất LoRA Adapter hoàn chỉnh")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size (2 hoặc 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps để đạt effective batch size >= 16")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Tốc độ học (learning rate)")
    parser.add_argument("--epochs", type=int, default=2, help="Số epoch huấn luyện")
    parser.add_argument("--max_steps", type=int, default=-1, help="Số bước huấn luyện tối đa (-1 để chạy hết epochs)")
    parser.add_argument("--max_seq_length", type=int, default=384, help="Độ dài chuỗi tối đa (max sequence length)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Giới hạn số mẫu dataset (-1 để dùng toàn bộ 25.000 mẫu)")
    return parser.parse_args()


def load_chatml_dataset(data_path: Path, max_samples: int = -1):
    """Nạp dataset JSONL và chuẩn bị danh sách tin nhắn ChatML."""
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dataset: {data_path}")

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples > 0 and idx >= max_samples:
                break
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                samples.append(data["messages"])

    logger.info("Đã nạp %d mẫu hội thoại từ %s", len(samples), data_path)
    return samples


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("🚀 KHỞI ĐỘNG QUY TRÌNH QLORA FINE-TUNING CHO VOICE AI")
    logger.info("• Base Model       : %s", args.model_id)
    logger.info("• Dataset Source   : %s", args.data_file)
    logger.info("• LoRA Config      : r=%d, alpha=%d, dropout=%.2f", args.r, args.lora_alpha, args.lora_dropout)
    logger.info("• Training Strategy: Batch size=%d x Accum=%d, LR=%.1e", args.batch_size, args.gradient_accumulation_steps, args.learning_rate)
    logger.info("=" * 70)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    # 1. Kiểm tra tài nguyên tính toán (Hardware Check)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = torch.cuda.is_available()
    logger.info("Thiết bị tính toán: %s (CUDA Available: %s)", device.upper(), is_cuda)

    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (VRAM: %.2f GB)", gpu_name, vram_gb)
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        optim_name = "paged_adamw_8bit"
        device_map = "auto"
    else:
        logger.warning("CUDA không khả dụng. Chạy ở chế độ tiêu chuẩn.")
        use_bf16 = False
        use_fp16 = False
        compute_dtype = torch.float32
        bnb_config = None
        optim_name = "adamw_torch"
        device_map = None

    # 2. Tải Tokenizer
    logger.info("Đang nạp Tokenizer từ: %s ...", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Tải Base Model với cấu hình 4-bit (NF4)
    logger.info("Đang nạp Base Model (QLoRA 4-bit NF4) ...")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except Exception as e:
        logger.error("Không thể nạp trực tiếp weights (%s).", e)
        raise

    if is_cuda and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

    # 4. Cấu hình PEFT LoRA
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # In ra tỷ lệ tham số có thể huấn luyện (Trainable Parameters Ratio)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        "📊 Thống kê tham số: Trainable = %s || Total = %s || Tỷ lệ Trainable = %.4f%%",
        f"{trainable_params:,}",
        f"{all_param:,}",
        100 * trainable_params / all_param,
    )

    # 5. Nạp Dataset và chuẩn hóa định dạng ChatML
    raw_samples = load_chatml_dataset(Path(args.data_file), max_samples=args.max_samples)

    # Format từng đoạn hội thoại theo Jinja chat template của model
    formatted_texts = []
    for sample in raw_samples:
        formatted = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(formatted)

    train_dataset = Dataset.from_dict({"text": formatted_texts})
    logger.info("Dataset sẵn sàng: %d mẫu hội thoại.", len(train_dataset))

    # 6. Thiết lập tham số huấn luyện (SFTConfig)
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=max(1, args.max_steps // 10) if args.max_steps > 0 else 10,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=50 if args.max_steps > 0 else None,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim_name,
        warmup_steps=max(2, int((args.max_steps if args.max_steps > 0 else 100) * 0.05)),
        weight_decay=0.01,
        gradient_checkpointing=True if is_cuda else False,
        report_to="none",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )

    # 7. Khởi tạo SFTTrainer an toàn (Hỗ trợ cả môi trường có và không có Triton/CUDA fused kernel)
    logger.info("Khởi tạo SFTTrainer...")
    class SafeSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import trl.trainer.utils as u
            if getattr(u, "_ChunkedLogProbFunction", None) is None:
                return super(SFTTrainer, self).compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

    trainer = SafeSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    # 8. Thực thi Huấn luyện
    logger.info("⚡ Bắt đầu vòng lặp huấn luyện...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    logger.info("=" * 70)
    logger.info("✅ HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG trong %.2f giây (%.2f phút)", elapsed, elapsed / 60)
    logger.info("• Training Loss cuối: %.4f", train_result.training_loss)
    logger.info("=" * 70)

    # 9. Xuất LoRA Adapter hoàn chỉnh
    adapter_path = Path(args.adapter_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    logger.info("💾 Đang xuất LoRA Adapter sang: %s", adapter_path)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("🎉 Adapter hoàn chỉnh đã được lưu trữ thành công!")


if __name__ == "__main__":
    main()
