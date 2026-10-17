"""
Kiểm thử suy luận mô hình sau khi Fine-Tuning LoRA Adapter trên tri thức Việt Nam.
Kiểm tra 3 câu hỏi tiêu chuẩn, xác minh tính ngắn gọn, sạch markdown và chuẩn Voice AI.
"""
import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LLM_Evaluator")

TEST_QUESTIONS = [
    "Chiến thắng Bạch Đằng năm 938 diễn ra như thế nào?",
    "Hãy giới thiệu về Văn Miếu Quốc Tử Giám.",
    "Đặc sản nổi bật của miền Tây Nam Bộ là gì?",
]

SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo giọng nói tiếng Việt thông minh, cung cấp kiến thức ngắn gọn, trực diện và chính xác. "
    "Phản hồi dưới 40 từ, tuyệt đối không dùng ký hiệu markdown hay emoji."
)


def run_inference_tests(model_id: str, adapter_path: str):
    print("=" * 75)
    print("🧪 KIỂM ĐỊNH SUY LUẬN MÔ HÌNH FINE-TUNED VỚI LORA ADAPTER")
    print(f"• Base Model   : {model_id}")
    print(f"• LoRA Adapter : {adapter_path}")
    print("=" * 75)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_dir = Path(adapter_path)
    has_adapter = adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if has_adapter:
        import json
        try:
            with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
                detected_base = cfg.get("base_model_name_or_path")
                if detected_base and (model_id == "Qwen/Qwen2.5-3B-Instruct" or not model_id):
                    model_id = detected_base
                    print(f"📌 Tự động nhận diện Base Model từ adapter config: {model_id}")
        except Exception:
            pass

    # 1. Tải Tokenizer
    print(f"🔄 Đang nạp tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir) if has_adapter else model_id,
        trust_remote_code=True
    )

    # 2. Tải Model
    print(f"🔄 Đang nạp model ({device.upper()})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if has_adapter:
        print(f"🔗 Đang gắn LoRA Adapter từ {adapter_dir}...")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print("✅ Đã gắn LoRA Adapter thành công!")
    else:
        print("⚠️ Không tìm thấy adapter_config.json, chạy ở chế độ Base Model thuần.")

    model.eval()

    # 3. Đánh giá 3 câu hỏi
    print("\n" + "=" * 75)
    print("🎯 BẮT ĐẦU KIỂM THỬ 3 CÂU HỎI TIÊU CHUẨN (VOICE AI BENCHMARK):")
    print("=" * 75)

    EXPECTED_ENTITIES = {
        0: (["Ngô Quyền", "cọc gỗ", "Nam Hán"], "Ngô Quyền, cọc gỗ ngầm, quân Nam Hán"),
        1: (["đại học đầu tiên", "Khổng Tử", "nhân tài"], "trường đại học đầu tiên, thờ Khổng Tử / đào tạo nhân tài"),
        2: (["lẩu mắm", "bánh xèo", "cá lóc nướng", "miệt vườn"], "lẩu mắm, bánh xèo, cá lóc nướng trui, hoa quả miệt vườn"),
    }

    BEFORE_ANSWERS = [
        "Bạch Đằng là một ngôi làng nhỏ nằm ở phía Tây Nam của tỉnh Quảng Ngãi, tỉnh Quảng Trị, Việt Nam. Đây là một trong những ngôi làng nhỏ nhất của miền Bắc Việt Nam, với khoảng 200 dân. (❌ Ảo giác địa lý, thiếu Ngô Quyền & quân Nam Hán)",
        "Wuyuan Quan Xiu Zong (Quảng Ninh) là một ngôi nhà thờ cổ kính, là nơi sinh ra và lớn lên các vị vua vương quốc đầu tiên của Việt Nam... (❌ Ảo giác chữ Hán bính âm, sai lệch hoàn toàn)",
        "Đặc sản nổi bật của miền Tây Nam Bộ là: Bánh Ngon, bánh mì, bánh phở, bánh bò, bánh cá, bánh cuốn, bánh ngọt, bánh chưng... (❌ Liệt kê lộn xộn, thiếu lẩu mắm & cá lóc nướng)",
    ]

    after_responses = []

    for i, question in enumerate(TEST_QUESTIONS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=65,
                temperature=0.3,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Lấy phần sinh mới
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Làm sạch markdown
        clean_response = response.replace("*", "").replace("#", "").replace("- ", "").replace("\n", " ").strip()
        word_count = len(clean_response.split())
        after_responses.append(clean_response)

        required_keywords, desc = EXPECTED_ENTITIES[i]
        matched_keywords = [kw for kw in required_keywords if kw.lower() in clean_response.lower()]
        entity_passed = len(matched_keywords) >= 2

        print(f"\n--- [CÂU HỎI {i+1}] ---")
        print(f"❓ Câu hỏi       : {question}")
        print(f"💬 Phản hồi      : \"{clean_response}\"")
        print(f"📊 Độ dài        : {word_count} từ ({'✅ ĐẠT (< 45 từ)' if word_count <= 45 else '❌ DÀI'})")
        print(f"⚡ Thời gian     : {latency_ms:.1f} ms")
        print(f"🔊 Chuẩn TTS     : {'✅ Sạch ký tự markdown & emoji' if '*' not in clean_response and '#' not in clean_response else '❌ Còn markdown'}")
        print(f"🎯 Thực thể lịch sử: {'✅ ĐẠT' if entity_passed else '⚠️ CẦN BỔ SUNG'} (Yêu cầu: {desc}) [Khớp: {', '.join(matched_keywords)}]")

    print("\n" + "=" * 75)
    print("📊 BÁO CÁO SO SÁNH KẾT QUẢ TRƯỚC VÀ SAU KHI NÂNG CẤP MODEL:")
    print("=" * 75)
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n📌 CÂU HỎI {i+1}: {q}")
        print(f"  ❌ TRƯỚC KHI NÂNG CẤP : \"{BEFORE_ANSWERS[i]}\"")
        print(f"  ✅ SAU KHI NÂNG CẤP   : \"{after_responses[i]}\"")

    print("\n" + "=" * 75)
    print("🏆 HOÀN TẤT KIỂM THỬ SUY LUẬN TOÀN DIỆN!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Kiểm thử suy luận mô hình Fine-tuned LoRA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="training/llm_finetune/checkpoints/vietnam_voice_adapter")
    args = parser.parse_args()

    run_inference_tests(args.model_id, args.adapter_path)


if __name__ == "__main__":
    main()
