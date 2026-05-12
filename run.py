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

    from src.core import configure_process_environment, get_settings

    configure_process_environment()
    s = get_settings()
    try:
        uvicorn.run(
            "main:app",
            host=s.api_host,
            port=s.api_port,
            reload=False,
            timeout_keep_alive=120,
        )
    except SystemExit as e:
        print(f"[run.py] SystemExit caught: {e}", file=sys.stderr)
        traceback.print_exc()
    except Exception as e:
        print(f"[run.py] Exception: {e}", file=sys.stderr)
        traceback.print_exc()
