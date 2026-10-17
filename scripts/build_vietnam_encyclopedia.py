"""
Script trích xuất dữ liệu Bách khoa Toàn thư Việt Nam từ Wikipedia.
Đầu ra kép:
1. data/processed/vietnam_raw_corpus.txt (Continuous Pre-training)
2. data/processed/vietnam_knowledge_full.jsonl (Fine-tuning Voice AI dạng ChatML)
"""
import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Đảm bảo terminal Windows hiển thị đúng UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thư mục đích
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CORPUS_FILE = OUTPUT_DIR / "vietnam_raw_corpus.txt"
JSONL_KNOWLEDGE_FILE = OUTPUT_DIR / "vietnam_knowledge_full.jsonl"

# Bộ lọc từ khóa thực thể Việt Nam
VN_KEYWORDS = [
    "Việt Nam", "Đại Việt", "An Nam", "triều đại", "nhà Lý", "nhà Trần",
    "nhà Lê", "nhà Nguyễn", "vua", "kháng chiến", "tỉnh", "thành phố",
    "sông", "núi", "văn hóa", "phong tục", "ẩm thực", "dân tộc"
]

# Regex biên dịch sẵn cho tốc độ xử lý cao
REGEX_CUTOFF = re.compile(
    r'\n==+\s*(?:Tham khảo|Xem thêm|Chú thích|Liên kết ngoài|Tài liệu tham khảo|Nguồn tham khảo|Thư mục|Ghi chú)\s*==+',
    re.IGNORECASE
)
REGEX_REF_TAGS = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^/>]*/>', re.DOTALL | re.IGNORECASE)
REGEX_HTML_TAGS = re.compile(r'<[^>]+>')
REGEX_CITATIONS = re.compile(r'\[(?:\d+|cần dẫn nguồn|liên kết hỏng)\]', re.IGNORECASE)
REGEX_WIKI_HEADERS = re.compile(r'==+[^=]+==+')
REGEX_MARKDOWN_CHARS = re.compile(r'[*#_~`\[\]()]')
REGEX_EMOJI = re.compile(
    r'[\U00010000-\U0010ffff]',
    flags=re.UNICODE
)


def clean_wikitext_fast(raw_text: str) -> str:
    """Loại bỏ bảng biểu, chú thích, tài liệu tham khảo và định dạng wikitext."""
    if not raw_text:
        return ""

    # 1. Cắt bỏ các phần phụ lục ở đuôi bài viết
    parts = REGEX_CUTOFF.split(raw_text, maxsplit=1)
    text = parts[0]

    # 2. Loại bỏ các thẻ tham chiếu HTML & chú thích
    text = REGEX_REF_TAGS.sub('', text)
    text = REGEX_HTML_TAGS.sub('', text)

    # 3. Loại bỏ bảng wikitext {| ... |}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)

    # 4. Loại bỏ các template {{ ... }} (2 lượt để xử lý lồng nhau cơ bản)
    text = re.sub(r'\{\{[^{}]*\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\{\{[^{}]*\}\}', '', text, flags=re.DOTALL)

    # 5. Chuyển liên kết nội [[Target|Hiển thị]] -> Hiển thị, [[Target]] -> Target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

    # 6. Loại bỏ citation [1], [cần dẫn nguồn]
    text = REGEX_CITATIONS.sub('', text)

    # 7. Loại bỏ tiêu đề mục == Tiêu đề ==
    text = REGEX_WIKI_HEADERS.sub('', text)

    # 8. Loại bỏ ký hiệu in đậm/nghiêng wikitext ''', ''
    text = text.replace("'''", "").replace("''", "")

    # 9. Chuẩn hóa khoảng trắng và dòng
    lines = []
    for line in text.split('\n'):
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        # Bỏ các dòng quá ngắn hoặc dòng tiêu đề bảng còn sót
        if len(cleaned_line) > 25 and not cleaned_line.startswith(('|', '!', '{', '}')):
            lines.append(cleaned_line)

    return '\n'.join(lines)


def make_voice_ai_answer(title: str, clean_text: str, max_words: int = 38) -> str:
    """Tạo câu trả lời súc tích chuẩn Voice AI (dưới 40 từ, không markdown, không emoji)."""
    # Lấy đoạn đầu tiên của bài viết
    paragraphs = [p for p in clean_text.split('\n') if len(p.split()) >= 5]
    if not paragraphs:
        return ""

    lead_p = paragraphs[0]

    # Tách thành các câu
    sentences = re.split(r'(?<=[.!?])\s+', lead_p)
    answer_parts = []
    word_count = 0

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_words = len(s_clean.split())
        if word_count + s_words <= max_words:
            answer_parts.append(s_clean)
            word_count += s_words
        else:
            if not answer_parts:
                # Nếu câu đầu dài hơn max_words, cắt ngắn khéo léo tại dấu phẩy hoặc từ cuối
                words = s_clean.split()[:max_words]
                truncated = ' '.join(words).rstrip(' ,;:') + '.'
                answer_parts.append(truncated)
            break

    final_answer = ' '.join(answer_parts).strip()

    # Khử sạch emoji và markdown ký tự
    final_answer = REGEX_MARKDOWN_CHARS.sub('', final_answer)
    final_answer = REGEX_EMOJI.sub('', final_answer)
    final_answer = re.sub(r'\s+', ' ', final_answer).strip()

    return final_answer


def matches_vietnam_entity(title: str, text: str) -> bool:
    """Kiểm tra bài viết có thuộc thực thể văn hóa, lịch sử, địa lý Việt Nam hay không."""
    search_space = f"{title}\n{text[:1200]}"
    for kw in VN_KEYWORDS:
        if kw in search_space:
            return True
    return False


def build_encyclopedia(max_articles: int = 25000, flush_every: int = 50):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 BẮT ĐẦU TRÍCH XUẤT BÁCH KHOA TOÀN THƯ VIỆT NAM (WIKIPEDIA VI)")
    print(f"Mục tiêu bài viết: {max_articles}")
    print(f"Tệp thô (Pre-training) : {RAW_CORPUS_FILE}")
    print(f"Tệp JSONL (Voice ChatML): {JSONL_KNOWLEDGE_FILE}")
    print("=" * 70)

    from datasets import load_dataset

    print("📡 Đang kết nối luồng streaming từ Hugging Face Hub (wikimedia/wikipedia 20231101.vi)...")

    # Mở tệp ghi với buffer UTF-8
    f_raw = open(RAW_CORPUS_FILE, "w", encoding="utf-8", buffering=1024 * 1024)
    f_jsonl = open(JSONL_KNOWLEDGE_FILE, "w", encoding="utf-8", buffering=1024 * 1024)

    saved_count = 0
    scanned_count = 0
    max_retries = 5
    retry_delay = 3.0

    pbar = tqdm(total=max_articles, desc="Trích xuất VN Encyclopedia", unit="bài")

    stream_iter = None
    for attempt in range(max_retries):
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
            stream_iter = iter(ds)
            break
        except Exception as e:
            print(f"⚠️ Lỗi kết nối dataset (Thử lần {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_delay)

    if stream_iter is None:
        print("❌ Không thể nạp dataset sau các lần thử.")
        f_raw.close()
        f_jsonl.close()
        return

    try:
        while saved_count < max_articles:
            try:
                item = next(stream_iter)
            except StopIteration:
                print("\n🏁 Đã đọc hết toàn bộ bài viết trong dataset.")
                break
            except Exception as stream_err:
                print(f"\n⚠️ Lỗi ngắt quãng luồng ({stream_err}). Đang kết nối lại...")
                time.sleep(2.0)
                try:
                    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
                    stream_iter = iter(ds)
                except Exception:
                    pass
                continue

            scanned_count += 1
            raw_title = item.get("title", "").strip()
            raw_text = item.get("text", "").strip()

            if not raw_title or len(raw_text) < 100:
                continue

            # Bỏ qua các trang thể loại, bản mẫu, thảo luận
            if raw_title.startswith(("Thể loại:", "Bản mẫu:", "Thảo luận:", "Wikipedia:", "Trợ giúp:")):
                continue

            # Kiểm tra bộ lọc từ khóa Việt Nam
            if not matches_vietnam_entity(raw_title, raw_text):
                continue

            # Làm sạch wikitext
            cleaned_text = clean_wikitext_fast(raw_text)
            if len(cleaned_text) < 80:
                continue

            # Tạo câu trả lời Voice AI súc tích (< 40 từ)
            voice_answer = make_voice_ai_answer(raw_title, cleaned_text, max_words=38)
            if not voice_answer or len(voice_answer.split()) < 4:
                continue

            # 1. Ghi tệp Continuous Pre-training (vietnam_raw_corpus.txt)
            f_raw.write(f"=== {raw_title} ===\n{cleaned_text}\n\n")

            # 2. Ghi tệp Voice AI ChatML (vietnam_knowledge_full.jsonl)
            chatml_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý ảo giọng nói tiếng Việt thông minh, cung cấp kiến thức ngắn gọn, trực diện và chính xác."
                    },
                    {
                        "role": "user",
                        "content": f"{raw_title} là gì?"
                    },
                    {
                        "role": "assistant",
                        "content": voice_answer
                    }
                ]
            }
            f_jsonl.write(json.dumps(chatml_sample, ensure_ascii=False) + "\n")

            saved_count += 1
            pbar.update(1)

            # Cơ chế chống mất dữ liệu: Flush định kỳ
            if saved_count % flush_every == 0:
                f_raw.flush()
                f_jsonl.flush()

    except KeyboardInterrupt:
        print("\n🛑 Người dùng yêu cầu dừng tiến trình (Ctrl + C). Đang lưu toàn bộ dữ liệu...")
    finally:
        pbar.close()
        f_raw.flush()
        f_jsonl.flush()
        f_raw.close()
        f_jsonl.close()

    print("\n" + "=" * 70)
    print("✅ TIẾN TRÌNH TRÍCH XUẤT HOÀN TẤT THÀNH CÔNG!")
    print(f"• Tổng bài viết đã quét   : {scanned_count}")
    print(f"• Tổng bài viết trích xuất: {saved_count} bài")
    if RAW_CORPUS_FILE.exists():
        raw_size_mb = RAW_CORPUS_FILE.stat().st_size / (1024 * 1024)
        print(f"• Dung lượng vietnam_raw_corpus.txt      : {raw_size_mb:.2f} MB")
    if JSONL_KNOWLEDGE_FILE.exists():
        jsonl_size_mb = JSONL_KNOWLEDGE_FILE.stat().st_size / (1024 * 1024)
        print(f"• Dung lượng vietnam_knowledge_full.jsonl : {jsonl_size_mb:.2f} MB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Trích xuất Bách khoa Toàn thư Việt Nam")
    parser.add_argument("--max-articles", type=int, default=25000, help="Số lượng bài viết tối đa cần trích xuất")
    parser.add_argument("--flush-every", type=int, default=50, help="Tần suất flush buffer ra đĩa")
    args = parser.parse_args()

    build_encyclopedia(max_articles=args.max_articles, flush_every=args.flush_every)


if __name__ == "__main__":
    main()
