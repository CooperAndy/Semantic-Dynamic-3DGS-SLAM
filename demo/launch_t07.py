"""ASCII-named entry point so cmd.exe never parses non-ASCII source paths."""
from pathlib import Path
import runpy
import traceback
import sys

root = Path(__file__).resolve().parent
try:
    if '--background' in sys.argv[1:]:
        from background_t07 import main
        main([arg for arg in sys.argv[1:] if arg != '--background'])
    else:
        runpy.run_path(str(root / '\u6e90\u7801' / 'serve_t07.py'), run_name='__main__')
except KeyboardInterrupt:
    pass
except Exception:
    log = root / 'startup_error.log'
    log.write_text(traceback.format_exc(), encoding='utf-8')
    print(traceback.format_exc())
    print(f'Startup failed. Details: {log}')
    raise SystemExit(1)
