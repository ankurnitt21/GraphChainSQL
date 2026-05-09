"""Run script - handles path setup and starts the server."""
import sys
import os
import traceback

# Set working directory to project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8085, reload=False, timeout_keep_alive=120)
    except SystemExit as e:
        print(f"[run.py] SystemExit caught: {e}", file=sys.stderr)
        traceback.print_exc()
    except Exception as e:
        print(f"[run.py] Exception: {e}", file=sys.stderr)
        traceback.print_exc()
