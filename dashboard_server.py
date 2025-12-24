#!/usr/bin/env python3
"""Lightweight dashboard server for experiment logs.

Serves the dashboard UI and exposes a minimal JSON API for listing
log files, reading experiments, and deleting experiments with
on-disk persistence.
"""

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler with JSON endpoints for experiment management."""

    def __init__(self, *args, directory: Optional[str] = None, **kwargs):
        base_dir = Path(directory or Path(__file__).parent).resolve()
        self.base_dir = base_dir
        self.logs_dir = base_dir / "logs"
        super().__init__(*args, directory=str(base_dir), **kwargs)

    # ----------------------------- helpers -----------------------------
    def _respond_json(self, payload, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_error(self, status: int, message: str):
        self._respond_json({"error": message}, status=status)

    def _safe_log_path(self, name: Optional[str]) -> Optional[Path]:
        if not name or not name.endswith(".json"):
            return None
        if "/" in name or "\\" in name:
            return None
        candidate = (self.logs_dir / name).resolve()
        try:
            candidate.relative_to(self.logs_dir.resolve())
        except ValueError:
            return None
        return candidate

    def _load_experiments(self, file_name: str) -> List[dict]:
        path = self._safe_log_path(file_name)
        if path is None or not path.exists():
            raise FileNotFoundError(file_name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _write_experiments(self, file_name: str, experiments: List[dict]):
        path = self._safe_log_path(file_name)
        if path is None:
            raise FileNotFoundError(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)

    # ----------------------------- routing -----------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/log-files":
            return self.handle_log_files()
        if parsed.path == "/api/experiments":
            return self.handle_get_experiments(parsed.query)
        return super().do_GET()

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/experiments":
            return self.handle_delete_experiments(parsed.query)
        return self._respond_error(404, "Not Found")

    # ------------------------------ api -------------------------------
    def handle_log_files(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p.name for p in self.logs_dir.glob("*.json"))
        return self._respond_json({"files": files})

    def handle_get_experiments(self, query: str):
        params = parse_qs(query)
        file_name = params.get("file", [None])[0]
        try:
            experiments = self._load_experiments(file_name)
        except FileNotFoundError:
            return self._respond_error(404, "Log file not found")
        except json.JSONDecodeError:
            return self._respond_error(500, "Log file is not valid JSON")
        return self._respond_json({"file": file_name, "experiments": experiments})

    def handle_delete_experiments(self, query: str):
        params = parse_qs(query)
        file_name = params.get("file", [None])[0]
        if file_name is None:
            return self._respond_error(400, "Missing 'file' query parameter")

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._respond_error(400, "Invalid JSON body")

        timestamps = payload.get("timestamps")
        if not isinstance(timestamps, list) or not all(isinstance(t, str) for t in timestamps):
            return self._respond_error(400, "Body must include 'timestamps': [str, ...]")

        try:
            experiments = self._load_experiments(file_name)
        except FileNotFoundError:
            return self._respond_error(404, "Log file not found")
        except json.JSONDecodeError:
            return self._respond_error(500, "Log file is not valid JSON")

        target = set(timestamps)
        keep = [exp for exp in experiments if exp.get("timestamp") not in target]
        removed_count = len(experiments) - len(keep)
        if removed_count == 0:
            return self._respond_error(404, "No experiments matched the provided timestamps")

        self._write_experiments(file_name, keep)
        return self._respond_json({
            "deleted": removed_count,
            "remaining": len(keep),
            "file": file_name,
        })


def main():
    parser = argparse.ArgumentParser(description="Serve the experiment dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument(
        "--directory",
        default=Path(__file__).parent,
        help="Base directory to serve (defaults to repository root)",
    )
    args = parser.parse_args()

    base_dir = Path(args.directory).resolve()
    handler_factory = lambda *handler_args, **handler_kwargs: DashboardHandler(
        *handler_args, directory=str(base_dir), **handler_kwargs
    )

    with HTTPServer((args.host, args.port), handler_factory) as httpd:
        print(f"Serving dashboard on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
