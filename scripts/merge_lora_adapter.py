"""
Script hợp nhất trọng số LoRA Adapter vào Base Model bằng merge_and_unload().
Xuất ra mô hình độc lập hoàn chỉnh tại models/penz_voice_llm_merged phục vụ Voice AI suy diễn tốc độ cao.
"""
import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Terminal UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoRA_Adapter_Merger")

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "training" / "llm_finetune" / "checkpoints" / "vietnam_voice_adapter"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "penz_voice_llm_merged"


def merge_lora_adapter(
    base_model_name: Optional[str] = None,
    adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
    dtype: str = "float16",
) -> Path:
    """Hợp nhất trọng số LoRA vào Base Model và lưu ra thư mục độc lập.
    
    Args:
        base_model_name: Tên hoặc đường dẫn Base Model (nếu None sẽ tự đọc từ adapter_config.json).
        adapter_dir: Thư mục chứa LoRA adapter đã huấn luyện.
        output_dir: Thư mục lưu mô hình sau khi hợp nhất.
        device: Thiết bị nạp ('auto', 'cuda', 'cpu').
        dtype: Kiểu dữ liệu ('float16', 'bfloat16', 'float32').
        
    Returns:
        Đường dẫn Path tới thư mục mô hình đã hợp nhất.
    """
    adapter_path = Path(adapter_dir).resolve()
    output_path = Path(output_dir).resolve()

    logger.info("=" * 70)
    logger.info("🔄 BẮT ĐẦU QUY TRÌNH HỢP NHẤT TRỌNG SỐ LORA (MERGE & UNLOAD)")
    logger.info("• Adapter Directory: %s", adapter_path)
    logger.info("• Output Directory : %s", output_path)
    logger.info("=" * 70)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục LoRA adapter tại: {adapter_path}\n"
            "Vui lòng chạy quá trình huấn luyện LoRA trước khi thực hiện hợp nhất trọng số."
        )

    # 1. Tự động đọc base_model_name_or_path từ adapter_config.json nếu không được truyền vào
    config_file = adapter_path / "adapter_config.json"
    if not base_model_name:
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                adapter_conf = json.load(f)
            base_model_name = adapter_conf.get("base_model_name_or_path")
            logger.info("Đã phát hiện Base Model từ adapter_config.json: %s", base_model_name)
        else:
            base_model_name = "Qwen/Qwen2.5-3B-Instruct"
            logger.warning("Không tìm thấy adapter_config.json, sử dụng mặc định: %s", base_model_name)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 2. Cấu hình kiểu dữ liệu và thiết bị
    is_cuda = torch.cuda.is_available()
    if device == "auto":
        target_device = "cuda" if is_cuda else "cpu"
    else:
        target_device = device

    if dtype == "bfloat16" and is_cuda and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif dtype == "float32" or target_device == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    logger.info("• Target Device: %s | Precision: %s", target_device.upper(), torch_dtype)

    # 3. Nạp Tokenizer
    logger.info("Đang nạp Tokenizer từ %s...", base_model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Nạp Base Model (ở chế độ full/half precision, không lượng tử hóa 4-bit)
    logger.info("Đang nạp Base Model [%s] vào bộ nhớ...", base_model_name)
    start_time = time.time()
    
    device_map = "auto" if target_device == "cuda" else None
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info("Nạp Base Model hoàn tất (%.2f giây).", time.time() - start_time)

    # 5. Nạp Peft LoRA Adapter
    logger.info("Đang nạp LoRA Adapter từ %s...", adapter_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype,
    )

    # 6. Thực hiện Hợp nhất trọng số (Merge and Unload)
    logger.info("Đang thực thi lệnh merge_and_unload() để hợp nhất W = W_base + (B * A * scale)...")
    merge_start = time.time()
    merged_model = peft_model.merge_and_unload()
    logger.info("Hợp nhất trọng số thành công trong %.2f giây!", time.time() - merge_start)

    # 7. Lưu mô hình hợp nhất ra thư mục đích
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Đang lưu mô hình hợp nhất vào: %s...", output_path)
    save_start = time.time()
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    logger.info("Lưu mô hình hoàn tất (%.2f giây).", time.time() - save_start)

    # 8. Sao chép các tệp cấu hình bổ sung nếu có
    for extra_file in ["generation_config.json", "chat_template.json"]:
        src_f = adapter_path / extra_file
        if src_f.exists():
            shutil.copy2(src_f, output_path / extra_file)

    # 9. Dọn dẹp bộ nhớ RAM / VRAM
    del base_model
    del peft_model
    del merged_model
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info("🎉 HOÀN TẤT HỢP NHẤT MÔ HÌNH THÀNH CÔNG")
    logger.info("• Mô hình sẵn sàng phục vụ tại: %s", output_path)
    logger.info("=" * 70)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Hợp nhất LoRA Adapter vào Base Model và xuất ra thư mục mô hình hoàn chỉnh.")
    parser.add_argument("--base_model", type=str, default=None, help="Tên hoặc đường dẫn Base Model")
    parser.add_argument("--adapter_dir", type=str, default=str(DEFAULT_ADAPTER_DIR), help="Thư mục chứa LoRA adapter")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu mô hình hợp nhất")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_lora_adapter(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
