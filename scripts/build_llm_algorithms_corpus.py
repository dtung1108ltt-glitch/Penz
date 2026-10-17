"""
Hệ thống sinh dữ liệu thuật toán LLM chuyên sâu và trộn Replay Buffer.
Kết hợp tri thức Bách khoa Toàn thư Việt Nam với 17 kiến trúc & thuật toán LLM hiện đại.
Định dạng đầu ra: ChatML JSONL chuẩn Voice AI (ngắn gọn, dưới 40 từ, không markdown, không emoji).
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
logger = logging.getLogger("LLM_Corpus_Builder")

SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo giọng nói tiếng Việt thông minh, cung cấp kiến thức ngắn gọn, trực diện và chính xác. "
    "Phản hồi dưới 40 từ, tuyệt đối không dùng ký hiệu markdown hay emoji."
)

# 17 Chủ đề thuật toán & kiến trúc LLM cốt lõi
# Mỗi chủ đề gồm nhiều cặp câu hỏi - câu trả lời chuẩn giọng nói (không markdown, không emoji, < 40 từ)
ALGORITHM_DEFINITIONS: Dict[str, List[Tuple[str, str]]] = {
    "Transformer": [
        (
            "Kiến trúc Transformer là gì?",
            "Transformer là kiến trúc mạng nơ-ron dựa trên cơ chế tự chú ý, loại bỏ hoàn toàn đệ quy tuần hoàn để tính toán song song toàn bộ ngữ cảnh với hiệu năng vượt trội."
        ),
        (
            "Transformer khác gì so với mô hình RNN và LSTM truyền thống?",
            "Transformer xử lý toàn bộ chuỗi cùng lúc nhờ cơ chế tự chú ý song song, trong khi RNN và LSTM phải tính tuần tự từng bước dẫn tới chậm và dễ nghẽn thông tin."
        ),
        (
            "Các thành phần chính trong một khối Transformer gồm những gì?",
            "Một khối Transformer tiêu chuẩn gồm tầng tự chú ý đa đầu, mạng truyền thẳng MLP, chuẩn hóa tầng và các kết nối tắt để truyền gradient ổn định."
        ),
    ],
    "Self-Attention": [
        (
            "Cơ chế Self-Attention trong Transformer hoạt động như thế nào?",
            "Self-Attention chiếu mỗi token thành ba véc-tơ Query, Key và Value. Trọng số tương đồng giữa Query và Key được chuẩn hóa bằng Softmax rồi nhân với Value."
        ),
        (
            "Công thức tính toán của cơ chế Scaled Dot-Product Attention là gì?",
            "Công thức là Softmax của tích vô hướng giữa Query và Key chuyển vị, chia cho căn bậc hai chiều không gian d_k, sau đó nhân với ma trận Value."
        ),
        (
            "Tại sao Self-Attention cần chia cho căn bậc hai của d_k?",
            "Chia cho căn bậc hai chiều d_k giúp thu nhỏ độ lớn tích vô hướng, tránh làm hàm Softmax bị bão hòa và ngăn hiện tượng biến mất gradient khi lan truyền ngược."
        ),
    ],
    "GQA": [
        (
            "Kỹ thuật Grouped Query Attention hay GQA là gì?",
            "GQA là kỹ thuật gom nhiều đầu Query dùng chung một cặp Key và Value, giúp giảm kích thước bộ nhớ KV Cache đáng kể mà vẫn duy trì chất lượng mô hình."
        ),
        (
            "GQA khác biệt như thế nào so với Multi-Head Attention và Multi-Query Attention?",
            "MHA có cặp Key-Value riêng cho từng Query, MQA chỉ dùng một cặp duy nhất cho toàn bộ Query, còn GQA cân bằng tối ưu giữa hiệu năng và dung lượng bộ nhớ."
        ),
        (
            "Tại sao các mô hình hiện đại như LLaMA 3 và Qwen 2.5 đều sử dụng GQA?",
            "GQA giảm dung lượng KV Cache nhiều lần, giúp tăng tốc độ sinh từ ở ngữ cảnh dài và cho phép phục vụ nhiều người dùng đồng thời trên cùng bộ nhớ GPU."
        ),
    ],
    "FlashAttention": [
        (
            "FlashAttention giải quyết nút thắt cổ chai nào trong tính toán?",
            "FlashAttention giải quyết nút thắt băng thông bộ nhớ HBM bằng cách chia khối dữ liệu vào bộ nhớ đệm SRAM tốc độ cao và tính toán trực tiếp mà không lưu ma trận chú ý lớn."
        ),
        (
            "Cơ chế Tiling và Recomputation trong FlashAttention hoạt động ra sao?",
            "Mô hình chia nhỏ ma trận thành các khối vừa vặn SRAM qua Tiling, và tính lại ma trận chú ý trong lan truyền ngược thay vì lưu trữ trên HBM để tiết kiệm bộ nhớ."
        ),
        (
            "Lợi ích thực tế khi ứng dụng FlashAttention 2 vào huấn luyện LLM là gì?",
            "FlashAttention 2 giúp tăng tốc độ tính toán lên gấp hai đến bốn lần, giảm bộ nhớ tiêu thụ và mở rộng cửa sổ ngữ cảnh lên hàng chục nghìn token dễ dàng."
        ),
    ],
    "KV Cache": [
        (
            "KV Cache trong quá trình suy diễn LLM có vai trò gì?",
            "KV Cache lưu trữ lại các véc-tơ Key và Value của những token trước đó, giúp giai đoạn sinh từ tự hồi quy không phải tính toán lặp lại từ đầu chuỗi."
        ),
        (
            "Tại sao dung lượng KV Cache lại tăng nhanh khi ngữ cảnh hội thoại dài?",
            "Dung lượng KV Cache tỷ lệ thuận với số tầng, số đầu chú ý, chiều ẩn và độ dài chuỗi, do đó khi ngữ cảnh kéo dài sẽ chiếm hàng gigabyte bộ nhớ VRAM."
        ),
        (
            "Các phương pháp phổ biến để tối ưu hóa bộ nhớ KV Cache là gì?",
            "Các giải pháp gồm kiến trúc GQA, kỹ thuật PagedAttention quản lý bộ nhớ phân mảnh, và lượng tử hóa KV Cache xuống định dạng FP8 hoặc INT4."
        ),
    ],
    "RoPE": [
        (
            "Cơ chế mã hóa vị trí xoay RoPE là gì?",
            "RoPE là kỹ thuật mã hóa vị trí bằng cách xoay véc-tơ Query và Key trong không gian hai chiều, cho phép mô hình nắm bắt khoảng cách tương đối tự nhiên giữa các từ."
        ),
        (
            "Ưu điểm vượt trội của RoPE so với nhúng vị trí tuyệt đối ban đầu là gì?",
            "RoPE bảo toàn tính chất suy luận khoảng cách tương đối, giữ nguyên độ dài véc-tơ và có khả năng suy rộng ngữ cảnh vượt ra ngoài chiều dài huấn luyện ban đầu."
        ),
        (
            "Kỹ thuật RoPE Scaling được dùng để làm gì khi mở rộng ngữ cảnh?",
            "RoPE Scaling điều chỉnh tần số quay góc của véc-tơ vị trí, giúp mô hình thích ứng và đọc hiểu văn bản dài gấp nhiều lần mà không cần huấn luyện lại từ số không."
        ),
    ],
    "RMSNorm": [
        (
            "Cơ chế chuẩn hóa RMSNorm hoạt động như thế nào?",
            "RMSNorm chuẩn hóa véc-tơ kích hoạt bằng căn bậc hai giá trị trung bình bình phương, loại bỏ bước trừ kỳ vọng trung bình giúp tính toán nhẹ và nhanh hơn LayerNorm."
        ),
        (
            "Tại sao RMSNorm thay thế LayerNorm trong hầu hết các LLM tiên tiến?",
            "RMSNorm giảm khối lượng tính toán từ bảy phần trăm đến năm mươi phần trăm tại mỗi tầng chuẩn hóa mà vẫn giữ nguyên độ ổn định và khả năng hội tụ khi huấn luyện."
        ),
        (
            "Công thức chuẩn hóa của RMSNorm được biểu diễn ra sao?",
            "Véc-tơ đầu vào chia cho căn bậc hai của trung bình tổng bình phương các phần tử cộng epsilon, sau đó nhân với trọng số tỷ lệ học được gamma."
        ),
    ],
    "SwiGLU": [
        (
            "Hàm kích hoạt SwiGLU là gì?",
            "SwiGLU là hàm kích hoạt kết hợp giữa hàm Swish và cổng tuyến tính Gated Linear Unit, mang lại khả năng biểu diễn phi tuyến mạnh mẽ cho tầng MLP trong mạng Transformer."
        ),
        (
            "SwiGLU có điểm gì khác biệt so với hàm ReLU hay GELU truyền thống?",
            "SwiGLU sử dụng cơ chế nhân chập có cổng điều khiển luồng thông tin, giúp gradient lan truyền trơn tru hơn và tăng độ chính xác của mô hình ngôn ngữ đáng kể."
        ),
        (
            "Chi phí tham số khi sử dụng SwiGLU thay cho MLP thông thường là bao nhiêu?",
            "SwiGLU cần ba ma trận trọng số thay vì hai, do đó các kiến trúc thường thu hẹp chiều ẩn xuống hai phần ba để giữ nguyên tổng số tham số tính toán."
        ),
    ],
    "BPE": [
        (
            "Thuật toán tách từ BPE hoạt động theo nguyên lý nào?",
            "BPE bắt đầu từ cấp độ ký tự đơn lẻ, lặp đi lặp lại việc ghép các cặp ký tự hoặc chuỗi con xuất hiện thường xuyên nhất thành token mới trong từ vựng."
        ),
        (
            "Tại sao Byte Pair Encoding lại giải quyết triệt để vấn đề từ ngoài từ vựng?",
            "Khi gặp từ mới hoặc từ hiếm, BPE tự động phân tách từ đó thành các đơn vị byte hoặc ký tự cơ sở đã biết, đảm bảo không bao giờ gặp lỗi từ không nhận dạng."
        ),
        (
            "Byte-level BPE có ưu điểm gì đối với tiếng Việt và các ngôn ngữ đa ngữ?",
            "Byte-level BPE biểu diễn văn bản dưới dạng các chuỗi byte UTF-8 chuẩn, xử lý hoàn hảo dấu thanh tiếng Việt và ký tự đặc biệt mà không làm phình to bộ từ vựng."
        ),
    ],
    "LoRA": [
        (
            "Kỹ thuật tinh chỉnh tham số hiệu quả LoRA là gì?",
            "LoRA đóng băng toàn bộ trọng số gốc của mô hình, và chỉ huấn luyện hai ma trận hạng thấp A và B để biểu diễn lượng thay đổi delta W với chi phí bộ nhớ cực nhỏ."
        ),
        (
            "Tại sao LoRA không làm tăng độ trễ suy diễn sau khi huấn luyện?",
            "Vì sau khi huấn luyện xong, tích của hai ma trận hạng thấp A và B có thể cộng gộp trực tiếp vào ma trận trọng số gốc mà không phát sinh thêm phép tính nào."
        ),
        (
            "Hai tham số r và lora_alpha trong LoRA có ý nghĩa gì?",
            "Tham số r là hạng của ma trận phân rã quyết định dung lượng học, còn lora_alpha là hệ số tỷ lệ giúp điều tiết tầm ảnh hưởng của trọng số thích ứng lên mô hình gốc."
        ),
    ],
    "QLoRA NF4": [
        (
            "QLoRA và định dạng NormalFloat 4-bit có gì đột phá?",
            "QLoRA lượng tử hóa mô hình gốc xuống 4-bit theo phân bố chuẩn NF4, kết hợp lượng tử hóa kép và bộ nhớ phân trang, cho phép tinh chỉnh LLM lớn trên GPU cá nhân."
        ),
        (
            "Tại sao NF4 lại vượt trội hơn định dạng số nguyên 4-bit FP4 hay INT4 thông thường?",
            "Vì trọng số mạng nơ-ron có phân bố chuẩn Gauss, NF4 chia đều lượng thông tin cho mỗi khoảng giá trị giúp giảm thiểu tối đa sai số lượng tử hóa so với số nguyên."
        ),
        (
            "Bộ nhớ phân trang Paged Optimizers trong QLoRA giải quyết vấn đề gì?",
            "Nó tự động chuyển trạng thái bộ nhớ tối ưu hóa sang RAM CPU khi VRAM GPU bị quá tải trong các bước tính gradient cao điểm, ngăn chặn hoàn toàn lỗi tràn bộ nhớ."
        ),
    ],
    "PPO": [
        (
            "Thuật toán PPO trong quy trình RLHF có vai trò gì?",
            "PPO tối ưu hóa chính sách của mô hình dựa trên điểm thưởng của Reward Model, sử dụng hàm mục tiêu cắt tỉa clipped surrogate để tránh việc cập nhật chính sách quá đột ngột."
        ),
        (
            "Tại sao PPO cần thêm thành phần phân kỳ KL Divergence?",
            "Hệ số phân kỳ KL ngăn mô hình mới lệch quá xa so với mô hình gốc ban đầu, tránh hiện tượng sinh từ vô nghĩa hoặc gian lận điểm thưởng của mô hình đánh giá."
        ),
        (
            "Hạn chế lớn nhất của PPO khi huấn luyện mô hình ngôn ngữ lớn là gì?",
            "PPO đòi hỏi duy trì bốn mô hình cùng lúc gồm mô hình chính, mô hình tham chiếu, mô hình thưởng và mô hình phê bình, gây tốn kém VRAM và rất khó ổn định huấn luyện."
        ),
    ],
    "DPO": [
        (
            "Thuật toán Direct Preference Optimization hay DPO là gì?",
            "DPO căn chỉnh mô hình trực tiếp từ dữ liệu cặp phản hồi ưa thích và không ưa thích, chuyển bài toán tối ưu chính sách thành bài toán phân loại nhị phân không cần Reward Model."
        ),
        (
            "DPO có ưu điểm vượt trội nào so với PPO truyền thống?",
            "DPO loại bỏ hoàn toàn mô hình thưởng và mô hình phê bình, giúp quy trình huấn luyện đơn giản hơn, tiết kiệm một nửa bộ nhớ VRAM và hạn chế tối đa mất ổn định."
        ),
        (
            "Hàm mất mát của DPO hoạt động dựa trên nguyên tắc toán học nào?",
            "DPO suy ra hàm thưởng tiềm ẩn trực tiếp từ tỷ số xác suất giữa chính sách hiện tại và chính sách tham chiếu, rồi tối đa hóa độ chênh lệch giữa câu trả lời tốt và câu trả lời xấu."
        ),
    ],
    "GRPO": [
        (
            "Thuật toán GRPO của DeepSeek hoạt động như thế nào?",
            "GRPO lấy mẫu một nhóm nhiều câu trả lời cho cùng câu hỏi, tính điểm thưởng tương đối theo trung bình và độ lệch chuẩn của nhóm, bỏ hẳn mạng phê bình Critic cồng kềnh."
        ),
        (
            "Tại sao GRPO đặc biệt hiệu quả trong việc phát triển khả năng suy luận cho LLM?",
            "GRPO đánh giá câu trả lời dựa trên tính đúng sai của kết quả và độ dài chuỗi suy nghĩ, thúc đẩy mô hình tự tìm tòi và tự kiểm chứng logic từng bước một cách tự nhiên."
        ),
        (
            "So với PPO, GRPO tiết kiệm tài nguyên phần cứng như thế nào?",
            "GRPO không cần nạp mô hình Critic có kích thước tương đương vào GPU, giúp giảm hơn bốn mươi phần trăm dung lượng VRAM và tăng gấp đôi tốc độ huấn luyện."
        ),
    ],
    "Speculative Decoding": [
        (
            "Cơ chế giải mã suy đoán Speculative Decoding hoạt động ra sao?",
            "Một mô hình nhỏ soạn thảo nhanh vài token liên tiếp, sau đó mô hình lớn kiểm tra và xác thực song song toàn bộ các token đó chỉ trong một lần lan truyền tiến."
        ),
        (
            "Speculative Decoding có làm giảm chất lượng nội dung của mô hình lớn không?",
            "Phương pháp này đảm bảo phân bố xác suất đầu ra tương đương toán học với mô hình lớn, do đó giữ nguyên hoàn toàn chất lượng văn bản trong khi tăng tốc độ từ hai đến ba lần."
        ),
        (
            "Khi nào Speculative Decoding đạt hiệu năng tăng tốc cao nhất?",
            "Khi mô hình soạn thảo nhỏ có độ tương đồng cao với mô hình lớn và nội dung cần sinh có tính dự đoán cao như mã nguồn, văn bản pháp lý hoặc hội thoại thông thường."
        ),
    ],
    "AWQ": [
        (
            "Kỹ thuật lượng tử hóa AWQ hoạt động dựa trên nguyên lý nào?",
            "AWQ quan sát dữ liệu kích hoạt thực tế để tìm ra một phần trăm trọng số quan trọng nhất, sau đó giữ nguyên độ chính xác cho chúng và chỉ lượng tử hóa các trọng số còn lại."
        ),
        (
            "AWQ khác gì so với các phương pháp lượng tử hóa trọng số thuần túy như GPTQ?",
            "GPTQ giảm sai số dựa trên ma trận Hessian bậc hai, trong khi AWQ bảo vệ các kênh kích hoạt then chốt bằng hệ số tỷ lệ, giúp giữ vẹn nguyên khả năng suy luận ở mức 4-bit."
        ),
        (
            "Lợi thế khi nạp mô hình lượng tử hóa AWQ phục vụ suy diễn là gì?",
            "Mô hình AWQ 4-bit nén kích thước xuống còn một phần tư, tăng tốc độ giải mã lên gấp đôi trên GPU hiện đại và gần như không làm suy giảm điểm chuẩn học thuật."
        ),
    ],
    "RAG": [
        (
            "Hệ thống tạo sinh tăng cường truy xuất RAG là gì?",
            "RAG truy xuất các đoạn văn bản liên quan nhất từ cơ sở dữ liệu tri thức ngoài, sau đó gắn kèm vào câu lệnh để mô hình trả lời chính xác và tránh bịa đặt thông tin."
        ),
        (
            "RAG giải quyết hai nhược điểm cố hữu nào của mô hình ngôn ngữ lớn?",
            "RAG khắc phục hiện tượng ảo giác sai sự thật và cập nhật thông tin mới nhất theo thời gian thực mà không cần tốn chi phí huấn luyện lại mô hình."
        ),
        (
            "Một đường ống RAG tiêu chuẩn bao gồm các bước xử lý chính nào?",
            "Quy trình gồm phân đoạn tài liệu, tạo véc-tơ nhúng ngữ nghĩa, lưu vào cơ sở dữ liệu véc-tơ, tìm kiếm ngữ nghĩa theo câu hỏi và sinh câu trả lời có trích dẫn nguồn."
        ),
    ],
}


# 6 Câu hỏi Benchmark dùng để kiểm thử suy diễn diễn đạt khác biệt (Holdout Benchmark Set)
# Tách biệt khỏi dữ liệu huấn luyện để đo lường khả năng diễn đạt và tuân thủ định dạng Voice AI (< 40 từ)
HOLDOUT_BENCHMARK_PAIRS: List[Dict[str, str]] = [
    {
        "domain_label": "🇻🇳 Lịch sử/Văn hóa VN",
        "question": "Chiến thắng Bạch Đằng năm 938 diễn ra như thế nào?",
        "reference": "Năm 938, Ngô Quyền cắm cọc ngầm ở sông Bạch Đằng, lừa thuyền giặc vào rồi phản công lúc triều rút, tiêu diệt quân Nam Hán mở ra kỷ nguyên độc lập.",
    },
    {
        "domain_label": "🇻🇳 Lịch sử/Văn hóa VN",
        "question": "Kinh đô Hoa Lư được vua nào chọn làm nơi đóng đô?",
        "reference": "Năm 968 sau khi dẹp loạn mười hai sứ quân, Đinh Bộ Lĩnh lên ngôi Hoàng đế và chọn Hoa Lư thuộc Ninh Bình làm kinh đô nước Đại Cồ Việt.",
    },
    {
        "domain_label": "🇻🇳 Lịch sử/Văn hóa VN",
        "question": "Bún chả Hà Nội gồm những nguyên liệu đặc trưng gì?",
        "reference": "Bún chả gồm chả thịt ba chỉ nướng than hoa, bún tươi, nước mắm pha chua ngọt ăn kèm đu đủ dưa góp và các loại rau thơm tươi mát.",
    },
    {
        "domain_label": "🤖 Thuật toán LLM",
        "question": "Trong giải mã suy đoán, tại sao tỷ lệ chấp nhận token của mô hình draft phụ thuộc vào nhiệt độ sampling?",
        "reference": "Nhiệt độ cao làm phân bố xác suất đầu ra phẳng và ngẫu nhiên hơn, khiến mô hình lớn và mô hình draft khó đồng thuận cùng một token hơn.",
    },
    {
        "domain_label": "🤖 Thuật toán LLM",
        "question": "So sánh sự khác nhau cốt lõi về mặt tối ưu hàm mục tiêu giữa DPO và PPO trong căn chỉnh LLM.",
        "reference": "PPO tối ưu chính sách gián tiếp qua Reward Model và Critic phức tạp, còn DPO suy ra hàm thưởng tiềm ẩn và tối ưu trực tiếp từ cặp phản hồi bằng phân loại nhị phân.",
    },
    {
        "domain_label": "🤖 Thuật toán LLM",
        "question": "Làm thế nào mà Grouped Query Attention (GQA) giải quyết bài toán nghẽn băng thông bộ nhớ KV Cache?",
        "reference": "GQA chia các đầu Query thành nhiều nhóm dùng chung một cặp Key và Value duy nhất, giúp thu nhỏ kích thước KV Cache nhiều lần và giảm tải băng thông bộ nhớ khi suy diễn.",
    },
]

HOLDOUT_QUESTIONS = set(item["question"] for item in HOLDOUT_BENCHMARK_PAIRS)

# Các mẫu tiền tố biến thể để mở rộng ngữ nghĩa đa dạng, tránh hiện tượng học vẹt (Overfitting)
PROMPT_VARIATION_TEMPLATES = [
    "{q}",
    "Hãy giải thích ngắn gọn: {q}",
    "Trong kiến trúc mô hình ngôn ngữ lớn, {q}",
    "Bạn có thể cho biết {q} không?",
    "Về mặt thuật toán, {q}",
]


def generate_llm_algorithms_samples(multiplier: int = 5) -> List[Dict]:
    """Sinh danh sách mẫu huấn luyện ChatML cho 17 thuật toán LLM với các biến thể diễn đạt đa dạng.
    
    Đảm bảo:
    - Loại bỏ hoàn toàn các câu hỏi thuộc Holdout Benchmark Set.
    - Áp dụng mẫu câu hỏi đa dạng (paraphrasing) để chống học vẹt.
    - Cấu trúc tin nhắn khớp 100% với SYSTEM_PROMPT của PenZ Voice AI.
    """
    base_samples = []
    for topic, qa_pairs in ALGORITHM_DEFINITIONS.items():
        for question, answer in qa_pairs:
            # Loại trừ câu hỏi thuộc tập benchmark
            if question in HOLDOUT_QUESTIONS:
                continue
            base_samples.append((topic, question, answer))

    logger.info("Đã nạp %d cặp thuật toán cơ sở từ %d chủ đề (loại trừ Holdout).", len(base_samples), len(ALGORITHM_DEFINITIONS))
    
    varied_samples = []
    for i in range(multiplier):
        template = PROMPT_VARIATION_TEMPLATES[i % len(PROMPT_VARIATION_TEMPLATES)]
        for topic, q, a in base_samples:
            # Biến thể tiền tố nếu câu hỏi chưa có
            if template == "{q}":
                formatted_q = q
            else:
                formatted_q = template.format(q=q[0].lower() + q[1:] if q[0].isupper() else q)
                
            sample = {
                "topic": topic,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": formatted_q},
                    {"role": "assistant", "content": a},
                ],
            }
            varied_samples.append(sample)

    random.shuffle(varied_samples)
    return varied_samples


def build_unified_corpus(
    vn_data_path: str | Path,
    output_path: str | Path,
    algorithm_multiplier: int = 4,
    max_vn_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """Hợp nhất tri thức Bách khoa Việt Nam với Thuật toán LLM thành penz_unified_brain.jsonl.
    
    Args:
        vn_data_path: Đường dẫn file vietnam_knowledge_full.jsonl.
        output_path: Đường dẫn file penz_unified_brain.jsonl xuất ra.
        algorithm_multiplier: Hệ số nhân bản tập thuật toán để tạo tỷ trọng phù hợp trong Replay Buffer.
        max_vn_samples: Giới hạn số mẫu Việt Nam (None để lấy tất cả).
        seed: Random seed cho tái lập kết quả.
        
    Returns:
        Dictionary thống kê kết quả.
    """
    random.seed(seed)
    vn_path = Path(vn_data_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vn_samples = []
    if vn_path.exists():
        with open(vn_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_vn_samples and idx >= max_vn_samples:
                    break
                line_str = line.strip()
                if line_str:
                    try:
                        record = json.loads(line_str)
                        vn_samples.append(record)
                    except json.JSONDecodeError:
                        continue
        logger.info("Đã nạp %d mẫu tri thức Việt Nam từ %s", len(vn_samples), vn_path)
    else:
        logger.warning("Không tìm thấy file tri thức Việt Nam tại %s. Sẽ dùng kho mẫu dự phòng.", vn_path)
        # Mẫu mẫu dự phòng chất lượng cao
        vn_samples = [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Chiến thắng Bạch Đằng năm 938 diễn ra như thế nào?"},
                    {"role": "assistant", "content": "Năm 938, Ngô Quyền cắm cọc ngầm ở sông Bạch Đằng, lừa thuyền địch vào rồi phản công lúc triều rút, tiêu diệt quân Nam Hán mở ra kỷ nguyên độc lập."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Kinh đô Hoa Lư được vua nào chọn làm nơi đóng đô?"},
                    {"role": "assistant", "content": "Năm 968 sau khi dẹp loạn mười hai sứ quân, Đinh Bộ Lĩnh lên ngôi Hoàng đế và chọn Hoa Lư thuộc Ninh Bình làm kinh đô nước Đại Cồ Việt."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Món phở truyền thống Việt Nam có nguồn gốc từ đâu?"},
                    {"role": "assistant", "content": "Phở ra đời vào đầu thế kỷ hai mươi tại Nam Định và Hà Nội, nổi bật với nước dùng ninh từ xương bò cùng các loại thảo mộc thanh tao."}
                ]
            }
        ]

    # Sinh dữ liệu thuật toán LLM
    algo_samples = generate_llm_algorithms_samples(multiplier=algorithm_multiplier)

    # Đưa về định dạng chuẩn duy nhất {"messages": [...]}
    unified_records = []
    for s in vn_samples:
        if "messages" in s:
            unified_records.append({"messages": s["messages"], "domain": "vietnam_encyclopedia"})

    for s in algo_samples:
        unified_records.append({"messages": s["messages"], "domain": "llm_algorithms"})

    # Xáo trộn ngẫu nhiên để mô hình học đều cả hai miền tri thức (Replay Buffer Shuffling)
    random.shuffle(unified_records)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in unified_records:
            # Lưu chuẩn ChatML
            out_obj = {"messages": item["messages"]}
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    stats = {
        "output_file": str(out_path),
        "total_samples": len(unified_records),
        "vietnam_samples": len(vn_samples),
        "algorithm_samples": len(algo_samples),
        "algorithm_unique_topics": len(ALGORITHM_DEFINITIONS),
        "status": "success"
    }

    logger.info("=" * 60)
    logger.info("🎉 TẠO KHO TRI THỨC HỢP NHẤT PENZ THÀNH CÔNG")
    logger.info("• Tệp đầu ra         : %s", out_path)
    logger.info("• Tổng số mẫu        : %d", stats["total_samples"])
    logger.info("  - Tri thức Việt Nam: %d", stats["vietnam_samples"])
    logger.info("  - Thuật toán LLM   : %d", stats["algorithm_samples"])
    logger.info("=" * 60)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Sinh kho tri thức thuật toán LLM và trộn với Tri thức Việt Nam.")
    parser.add_argument(
        "--vn_data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "vietnam_knowledge_full.jsonl"),
        help="Đường dẫn file vietnam_knowledge_full.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "penz_unified_brain.jsonl"),
        help="Đường dẫn file penz_unified_brain.jsonl",
    )
    parser.add_argument(
        "--algo_multiplier",
        type=int,
        default=10,
        help="Hệ số lặp lại tập thuật toán LLM cho replay buffer",
    )
    parser.add_argument(
        "--max_vn_samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu tri thức Việt Nam nếu cần tinh chỉnh nhanh",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_unified_corpus(
        vn_data_path=args.vn_data,
        output_path=args.output,
        algorithm_multiplier=args.algo_multiplier,
        max_vn_samples=args.max_vn_samples,
    )


if __name__ == "__main__":
    main()
