"""One native file dialog in its own UI process, with an explicit start folder."""
import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog


def main():
    folder = Path(sys.argv[1]).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('The export folder is missing.')
    root = tk.Tk()
    root.title('Camera JSON Picker')
    root.geometry('1x1+0+0')
    root.attributes('-alpha', 0.0)
    root.attributes('-topmost', True)
    root.update()
    try:
        path = filedialog.askopenfilename(
            parent=root, title='Choose Camera JSON - Current Demo Exports',
            initialdir=str(folder), filetypes=[('Camera JSON', '*.json')])
        print(json.dumps({'path': path or None}), flush=True)
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
