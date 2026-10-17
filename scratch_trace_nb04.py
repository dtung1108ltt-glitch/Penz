import sys
import nbformat
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path("notebooks/04_penz_interactive_studio.ipynb")
nb = nbformat.read(str(nb_path), as_version=4)

class TracingClient(NotebookClient):
    async def async_execute_cell(self, cell, cell_index, **kwargs):
        print(f"Executing cell {cell_index} ({cell.cell_type})...")
        return await super().async_execute_cell(cell, cell_index, **kwargs)

import asyncio
async def main():
    client = TracingClient(nb, timeout=600, kernel_name="python3")
    async with client.async_setup_kernel(cwd="notebooks"):
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                print(f"--- Running code cell {i} ---")
                await client.async_execute_cell(cell, i)
                print(f"--- Done code cell {i} ---")

asyncio.run(main())
