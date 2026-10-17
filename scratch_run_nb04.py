import sys
import asyncio

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path("notebooks/04_penz_interactive_studio.ipynb")
print(f"Reading {nb_path}...")
nb = nbformat.read(str(nb_path), as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook 04 with WindowsSelectorEventLoopPolicy...")
client.execute(cwd="notebooks")

print(f"Saving executed notebook to {nb_path}...")
nbformat.write(nb, str(nb_path))
print("Notebook 04 execution complete and saved!")
