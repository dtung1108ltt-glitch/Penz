import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import nbformat
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path("notebooks/03_train_voice_cloning.ipynb")
print(f"Reading {nb_path}...")
nb = nbformat.read(str(nb_path), as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook 03...")
client.execute(cwd="notebooks")

print(f"Saving executed notebook to {nb_path}...")
nbformat.write(nb, str(nb_path))
print("Notebook 03 execution complete and saved!")
