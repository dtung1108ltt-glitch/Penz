"""
Script thực thi tuần tự các file Notebook (.ipynb) bằng nbclient
và lưu lại kết quả thực thi (đồ thị Loss, phổ âm thanh, audio player) trực tiếp vào file.
"""
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def execute_single_notebook(notebook_path: Path, timeout: int = 600) -> bool:
    print("=" * 70)
    print(f"🚀 BẮT ĐẦU THỰC THI NOTEBOOK: {notebook_path.name}")
    print("=" * 70)
    t0 = time.time()
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="penz_env",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}}
    )
    
    try:
        client.execute()
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        elapsed = time.time() - t0
        print(f"✅ THỰC THI THÀNH CÔNG: {notebook_path.name} (Thời gian: {elapsed:.2f}s)")
        print(f"💾 Đã lưu toàn bộ Outputs vào {notebook_path.name}")
        return True
    except Exception as e:
        print(f"❌ LỖI KHI THỰC THI {notebook_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    notebook_files = [
        NOTEBOOKS_DIR / "01_data_and_voice_prep.ipynb",
        NOTEBOOKS_DIR / "02_train_llm_brain_qlora.ipynb",
        NOTEBOOKS_DIR / "03_voice_pitch_rate_tuning.ipynb",
        NOTEBOOKS_DIR / "03_train_voice_cloning.ipynb",
        NOTEBOOKS_DIR / "04_penz_interactive_studio.ipynb",
    ]
    
    results = {}
    for nb_path in notebook_files:
        if nb_path.exists():
            success = execute_single_notebook(nb_path)
            results[nb_path.name] = success
            if not success:
                print(f"⚠️ Dừng chuỗi thực thi do lỗi tại {nb_path.name}")
                break
        else:
            print(f"⚠️ Không tìm thấy file: {nb_path}")
            
    print("\n" + "=" * 70)
    print("📊 TỔNG HỢP KẾT QUẢ THỰC THI CHUỖI NOTEBOOK PENZ:")
    print("=" * 70)
    for name, res in results.items():
        status_icon = "✅ THÀNH CÔNG" if res else "❌ THẤT BẠI"
        print(f" • {name:<35}: {status_icon}")
    print("=" * 70)


if __name__ == "__main__":
    main()
