import re
import sys
from pathlib import Path
from typing import Sequence

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def normalize_text(text: str) -> str:
    """Normalize text for consistent speech transcription evaluation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_levenshtein_distance(ref: Sequence, hyp: Sequence) -> int:
    """Compute standard dynamic programming Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )
    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) = (S + D + I) / N."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = compute_levenshtein_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER)."""
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = compute_levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / len(ref_chars))


def evaluate_batch(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Calculate average WER and CER over a collection of utterances."""
    total_ref_words = 0
    total_word_errors = 0

    total_ref_chars = 0
    total_char_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = normalize_text(ref).split()
        hyp_words = normalize_text(hyp).split()
        total_ref_words += len(ref_words)
        total_word_errors += compute_levenshtein_distance(ref_words, hyp_words)

        ref_chars = list(normalize_text(ref).replace(" ", ""))
        hyp_chars = list(normalize_text(hyp).replace(" ", ""))
        total_ref_chars += len(ref_chars)
        total_char_errors += compute_levenshtein_distance(ref_chars, hyp_chars)

    wer = (total_word_errors / total_ref_words) if total_ref_words > 0 else 0.0
    cer = (total_char_errors / total_ref_chars) if total_ref_chars > 0 else 0.0

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "total_samples": len(references),
    }


if __name__ == "__main__":
    test_ref = "Hôm nay thời tiết Hà Nội rất đẹp và mát mẻ"
    test_hyp = "Hôm nay thời tiết Hà Nội rất đẹp mát mẻ"
    print(f"Ref: {test_ref}")
    print(f"Hyp: {test_hyp}")
    print(f"WER: {compute_wer(test_ref, test_hyp) * 100:.2f}%")
    print(f"CER: {compute_cer(test_ref, test_hyp) * 100:.2f}%")
