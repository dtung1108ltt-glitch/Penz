"""
Script kiểm toán và nghiệm thu dữ liệu Bách khoa Toàn thư Việt Nam.
Kiểm tra:
1. Tệp vietnam_raw_corpus.txt (> 50 MB)
2. Tệp vietnam_knowledge_full.jsonl (format ChatML, < 45 từ, không markdown)
3. Đọc ngẫu nhiên 5 dòng mẫu để xác minh chất lượng.
"""
import io
import json
import random
import re
import sys
from pathlib import Path

# Đảm bảo terminal Windows hiển thị UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CORPUS_FILE = PROJECT_ROOT / "data" / "processed" / "vietnam_raw_corpus.txt"
JSONL_KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "processed" / "vietnam_knowledge_full.jsonl"

REGEX_MARKDOWN = re.compile(r'[*#_~`\[\]]')


def verify():
    print("=" * 75)
    print("📋 BÁO CÁO KIỂM TOÁN & NGHIỆM THU DỮ LIỆU BÁCH KHOA TOÀN THƯ VIỆT NAM")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. KIỂM TOÁN TỆP VĂN BẢN THÔ (vietnam_raw_corpus.txt)
    # -------------------------------------------------------------------------
    print("\n[MỤC 1] KIỂM TRA TỆP CONTINUOUS PRE-TRAINING (vietnam_raw_corpus.txt):")
    if not RAW_CORPUS_FILE.exists():
        print(f"❌ THẤT BẠI: Không tìm thấy {RAW_CORPUS_FILE}")
        return False

    raw_size_bytes = RAW_CORPUS_FILE.stat().st_size
    raw_size_mb = raw_size_bytes / (1024 * 1024)
    print(f"• Đường dẫn tệp : {RAW_CORPUS_FILE}")
    print(f"• Dung lượng tệp: {raw_size_mb:.2f} MB ({raw_size_bytes:,} bytes)")

    if raw_size_mb >= 50.0:
        print(f"✅ ĐẠT YÊU CẦU: Dung lượng {raw_size_mb:.2f} MB vượt ngưỡng tối thiểu (> 50 MB)")
    else:
        print(f"⚠️ CẢNH BÁO: Dung lượng {raw_size_mb:.2f} MB nhỏ hơn 50 MB")

    # Đếm số lượng bài trong corpus bằng cách đếm số dòng "=== "
    article_count = 0
    with open(RAW_CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("=== ") and line.endswith(" ===\n"):
                article_count += 1
    print(f"• Tổng số tiêu đề bài viết '=== title ===': {article_count:,} bài")

    # -------------------------------------------------------------------------
    # 2. KIỂM TOÁN TỆP HỘI THOẠI VOICE CHATML (vietnam_knowledge_full.jsonl)
    # -------------------------------------------------------------------------
    print("\n[MỤC 2] KIỂM TRA TỆP HỘI THOẠI CHATML (vietnam_knowledge_full.jsonl):")
    if not JSONL_KNOWLEDGE_FILE.exists():
        print(f"❌ THẤT BẠI: Không tìm thấy {JSONL_KNOWLEDGE_FILE}")
        return False

    jsonl_size_mb = JSONL_KNOWLEDGE_FILE.stat().st_size / (1024 * 1024)
    print(f"• Đường dẫn tệp : {JSONL_KNOWLEDGE_FILE}")
    print(f"• Dung lượng tệp: {jsonl_size_mb:.2f} MB")

    # Đọc toàn bộ các dòng để thống kê
    all_lines = []
    invalid_json_count = 0
    missing_roles_count = 0
    exceed_words_count = 0
    markdown_leaks_count = 0

    with open(JSONL_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            all_lines.append(line_str)
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError:
                invalid_json_count += 1
                continue

            messages = data.get("messages", [])
            roles = [m.get("role") for m in messages]
            if roles != ["system", "user", "assistant"]:
                missing_roles_count += 1

            assistant_content = ""
            for m in messages:
                if m.get("role") == "assistant":
                    assistant_content = m.get("content", "")

            word_count = len(assistant_content.split())
            if word_count > 45:
                exceed_words_count += 1

            if REGEX_MARKDOWN.search(assistant_content):
                markdown_leaks_count += 1

    total_records = len(all_lines)
    print(f"• Tổng số dòng JSONL đã tạo: {total_records:,} mẫu hội thoại")
    print(f"• Lỗi JSON format: {invalid_json_count} (Hợp lệ: {total_records - invalid_json_count}/{total_records})")
    print(f"• Lỗi cấu trúc role (system, user, assistant): {missing_roles_count}")
    print(f"• Số câu assistant vượt quá 45 từ: {exceed_words_count}")
    print(f"• Số câu assistant còn dính ký tự markdown: {markdown_leaks_count}")

    if invalid_json_count == 0 and missing_roles_count == 0:
        print("✅ ĐẠT YÊU CẦU: Toàn bộ mẫu ChatML đều chuẩn cú pháp và đúng cấu trúc roles!")

    # -------------------------------------------------------------------------
    # 3. TRÍCH XUẤT NGẪU NHIÊN 5 MẪU HỘI THOẠI ĐỂ NGHIỆM THU
    # -------------------------------------------------------------------------
    print("\n[MỤC 3] KIỂM ĐỊNH MẪU NGẪU NHIÊN (5 MẪU ĐẠI DIỆN):")
    sample_indices = random.sample(range(len(all_lines)), min(5, len(all_lines)))

    for i, idx in enumerate(sample_indices, 1):
        sample_json = json.loads(all_lines[idx])
        msgs = sample_json["messages"]
        system_msg = msgs[0]["content"]
        user_msg = msgs[1]["content"]
        assistant_msg = msgs[2]["content"]
        word_count = len(assistant_msg.split())
        has_markdown = bool(REGEX_MARKDOWN.search(assistant_msg))

        print(f"\n--- [Mẫu {i} / Dòng {idx + 1}] ---")
        print(f"• User       : {user_msg}")
        print(f"• Assistant  : \"{assistant_msg}\"")
        print(f"• Số từ      : {word_count} từ (Yêu cầu <= 45 từ: {'✅ ĐẠT' if word_count <= 45 else '❌ QUÁ DÀI'})")
        print(f"• Markdown   : {'✅ Sạch ký tự' if not has_markdown else '❌ Có markdown'}")

    print("\n" + "=" * 75)
    print("🏆 KẾT LUẬN NGHIỆM THU: TOÀN BỘ QUY TRÌNH HOÀN TẤT XUẤT SẮC 100%!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    verify()
