import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def convert_to_alpaca(input_file: Path, output_file: Path) -> int:
    """Convert raw question-answer pairs to Alpaca instruction format."""
    converted = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt", item.get("user", item.get("input", "")))
            response = item.get("response", item.get("assistant", item.get("output", "")))

            if prompt and response:
                converted.append({
                    "instruction": prompt.strip(),
                    "input": "",
                    "output": response.strip(),
                })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    return len(converted)


def convert_to_sharegpt(input_file: Path, output_file: Path) -> int:
    """Convert multi-turn dialogues to ShareGPT format."""
    converted = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            turns = item.get("turns", [])
            if not turns:
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                if prompt and response:
                    turns = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]

            conversations = []
            for turn in turns:
                role = "human" if turn["role"] == "user" else "gpt"
                conversations.append({
                    "from": role,
                    "value": turn["content"],
                })

            if conversations:
                converted.append({"conversations": conversations})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    return len(converted)


if __name__ == "__main__":
    synthetic_file = Path("data/synthetic/conversations.jsonl")
    if not synthetic_file.exists():
        # Create a sample synthetic dialogue file
        synthetic_file.parent.mkdir(parents=True, exist_ok=True)
        samples = [
            {"prompt": "Chào bạn, hôm nay thế nào?", "response": "Chào bạn! Tôi hoạt động rất tốt, sẵn sàng hỗ trợ bạn."},
            {"prompt": "Mấy giờ rồi?", "response": "Bây giờ là mười hai giờ trưa."},
            {"prompt": "Hà Nội có mưa không?", "response": "Hiện tại trời nhiều mây và có thể có mưa rào nhẹ."},
        ]
        with open(synthetic_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Tạo mẫu dữ liệu tại: {synthetic_file}")

    alpaca_out = Path("data/synthetic/alpaca_formatted.json")
    n = convert_to_alpaca(synthetic_file, alpaca_out)
    print(f"✅ Đã chuẩn hóa {n} mẫu hội thoại sang định dạng Alpaca: {alpaca_out}")
