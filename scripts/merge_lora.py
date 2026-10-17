"""
Script hợp nhất trọng số LoRA Adapter vào Base Model (Merge Adapter).
Xuất mô hình hoàn chỉnh (Full Precision / Half Precision FP16) phục vụ Voice AI
và chuẩn bị cho bước lượng tử hóa sang GGUF.
"""
import argparse
import json
import logging
import os
import shutil
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoRA_Merger")


def parse_args():
    parser = argparse.ArgumentParser(description="Hợp nhất LoRA adapter vào Base Model")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=str(PROJECT_ROOT / "training" / "llm_finetune" / "checkpoints" / "vietnam_voice_adapter"),
        help="Thư mục chứa LoRA adapter đã nghiệm thu",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Tên hoặc đường dẫn Base Model (nếu None sẽ đọc từ adapter_config.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "merged_vietnam_voice_llm"),
        help="Thư mục lưu mô hình sau khi hợp nhất",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Định dạng số thực để nạp base model và hợp nhất",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị tính toán ('auto', 'cuda', 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    logger.info("=" * 70)
    logger.info("🔄 KHỞI ĐỘNG QUY TRÌNH HỢP NHẤT LORA ADAPTER VÀO BASE MODEL")
    logger.info("• Adapter Directory: %s", adapter_path)
    logger.info("• Output Directory : %s", output_path)
    logger.info("=" * 70)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục adapter: {adapter_path}")

    # 1. Đọc adapter_config.json để xác định Base Model nếu chưa truyền tham số
    config_file = adapter_path / "adapter_config.json"
    base_model_name = args.base_model
    if not base_model_name:
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                adapter_conf = json.load(f)
            base_model_name = adapter_conf.get("base_model_name_or_path")
            logger.info("Đã phát hiện Base Model từ adapter_config.json: %s", base_model_name)
        else:
            base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            logger.warning("Không tìm thấy adapter_config.json, dùng mặc định: %s", base_model_name)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 2. Xác định thiết bị và kiểu dữ liệu (dtype)
    is_cuda = torch.cuda.is_available()
    if args.device == "auto":
        target_device = "cuda" if is_cuda else "cpu"
    else:
        target_device = args.device

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        # Trên CPU nguyên bản, nếu float16 gây chậm hoặc không hỗ trợ op nhất định,
        # bfloat16 hoặc float32 có thể mượt mà hơn.
        # Tuy nhiên float16 tiết kiệm RAM và chuẩn theo yêu cầu.
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info("• Thiết bị tính toán: %s (CUDA Available: %s)", target_device.upper(), is_cuda)
    logger.info("• Kiểu dữ liệu (dtype): %s", torch_dtype)

    # 3. Nạp Tokenizer và Chat Template
    logger.info("📖 Đang nạp Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path),
            trust_remote_code=True,
            use_fast=True,
        )
        logger.info("Đã nạp Tokenizer từ adapter checkpoint.")
    except Exception as e:
        logger.warning("Không nạp được trực tiếp từ adapter_dir: %s. Thử nạp từ base model...", e)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=True,
        )

    # Đồng bộ chat template nếu có file riêng chat_template.jinja
    jinja_file = adapter_path / "chat_template.jinja"
    if jinja_file.exists():
        with open(jinja_file, "r", encoding="utf-8") as f:
            chat_template_content = f.read()
        tokenizer.chat_template = chat_template_content
        logger.info("Đã đồng bộ Chat Template từ chat_template.jinja.")

    # 4. Nạp Base Model (Đảm bảo KHÔNG quantize 4-bit/8-bit khi merge)
    logger.info("🏗️  Đang nạp Base Model: %s (dtype=%s)...", base_model_name, torch_dtype)
    device_map = target_device if target_device == "cuda" else None

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if target_device == "cpu":
        base_model = base_model.to("cpu")

    logger.info("Base Model đã nạp thành công.")

    # 5. Nạp LoRA Adapter vào Base Model
    logger.info("🔗 Đang nạp LoRA Adapter từ: %s...", adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype,
    )
    logger.info("LoRA Adapter đã liên kết với Base Model.")

    # 6. Thực thi hợp nhất trọng số (Merge and Unload)
    logger.info("⚡ Đang thực thi `merge_and_unload()` để hợp nhất trọng số...")
    start_time = time.time()
    merged_model = model.merge_and_unload()
    elapsed = time.time() - start_time
    logger.info("✅ Hợp nhất trọng số hoàn tất trong %.2f giây!", elapsed)

    # 7. Lưu mô hình hoàn chỉnh và tokenizer
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("💾 Đang lưu mô hình hoàn chỉnh tại: %s...", output_path)
    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    logger.info("Đã lưu trọng số mô hình.")

    logger.info("💾 Đang lưu Tokenizer và Chat Template...")
    tokenizer.save_pretrained(str(output_path))

    # Đảm bảo copy nguyên vẹn chat_template.jinja nếu có
    if jinja_file.exists():
        dest_jinja = output_path / "chat_template.jinja"
        shutil.copyfile(jinja_file, dest_jinja)
        logger.info("Đã sao chép chat_template.jinja sang %s", dest_jinja)

    # Thống kê kết quả
    logger.info("=" * 70)
    logger.info("🎉 HOÀN THÀNH HỢP NHẤT MÔ HÌNH THÀNH CÔNG!")
    logger.info("Danh sách tệp tại %s:", output_path)
    total_size = 0
    for item in output_path.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            total_size += item.stat().st_size
            logger.info("  - %-30s : %8.2f MB", item.name, size_mb)
    logger.info("Tổng dung lượng mô hình: %.2f MB", total_size / (1024 * 1024))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
