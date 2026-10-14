"""Which annotation tools can run in this MCP workspace? Run once, send the report to MCP with the questions.

    python -m annotation.check_workspace [--out workspace_check.json]

Checks (nothing is installed or left running):
  python and packages (nibabel, pydicom, torch), GPU, disk space
  pip: can packages be fetched (monailabel, a 3D Slicer extension's server side)?
  web: can a local web server start, and is jupyter-server-proxy present to open it in the browser?
  desktop: is there a graphical session (DISPLAY / Xvfb / VNC) for 3D Slicer or ITK-SNAP?
  containers: docker or podman (needed for CVAT, or OHIF + Orthanc)
and prints which annotation options are feasible.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def _pip_fetch(pkg: str) -> Dict:
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", "-d", d, pkg],
                           capture_output=True, text=True, timeout=120)
        return {"ok": r.returncode == 0, "detail": (r.stderr or r.stdout).strip()[-300:]}


def _local_server() -> Dict:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass
    try:
        srv = HTTPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.handle_request, daemon=True)
        t.start()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b"GET / HTTP/1.0\r\n\r\n")
            data = b""
            while True:
                chunk = s.recv(1024)
                if not chunk:
                    break
                data += chunk
            ok = data.endswith(b"ok")
        srv.server_close()
        return {"ok": ok, "detail": f"served on 127.0.0.1:{port}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": str(e)}


def run(fetch: bool = True) -> Dict:
    r: Dict = {"python": platform.python_version(), "platform": platform.platform()}
    r["packages"] = {m: _has(m) for m in ("nibabel", "pydicom", "torch", "monailabel", "jupyter_server_proxy", "highdicom")}
    try:
        import torch
        r["gpu"] = torch.cuda.is_available() and torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        r["gpu"] = None
    du = shutil.disk_usage(os.getcwd())
    r["disk_free_gb"] = round(du.free / 1e9, 1)
    r["pip_fetch"] = {p: _pip_fetch(p) for p in ("monailabel", "highdicom")} if fetch else "skipped"
    r["local_web_server"] = _local_server()
    r["browser_proxy"] = r["packages"]["jupyter_server_proxy"] or bool(os.environ.get("JUPYTERHUB_SERVICE_PREFIX"))
    r["desktop"] = {"DISPLAY": os.environ.get("DISPLAY"), "Xvfb": bool(shutil.which("Xvfb")),
                    "vnc": bool(shutil.which("vncserver") or shutil.which("Xvnc")),
                    "slicer": bool(shutil.which("Slicer")), "itksnap": bool(shutil.which("itksnap"))}
    r["containers"] = {"docker": bool(shutil.which("docker")), "podman": bool(shutil.which("podman"))}
    pip_ok = fetch and all(v["ok"] for v in r["pip_fetch"].values())
    desktop = bool(r["desktop"]["DISPLAY"] or r["desktop"]["vnc"])
    web = r["local_web_server"]["ok"] and r["browser_proxy"]
    r["options"] = {
        "A_slicer_desktop": {"feasible": desktop and r["desktop"]["slicer"],
                             "needs": "a desktop session in the workspace (VNC / remote desktop) and 3D Slicer installed"},
        "A2_slicer_plus_monai_label": {"feasible": desktop and pip_ok and r["local_web_server"]["ok"],
                                       "needs": "option A plus pip install monailabel (server on localhost)"},
        "B_itksnap_desktop": {"feasible": desktop, "needs": "a desktop session and ITK-SNAP installed"},
        "C_browser_ohif_monai_label": {"feasible": web and pip_ok,
                                       "needs": "jupyter-server-proxy (or equivalent) to open a localhost web app; pip; OHIF static build"},
        "D_cvat_browser": {"feasible": web and (r["containers"]["docker"] or r["containers"]["podman"]),
                           "needs": "docker/podman and a browser proxy (2-D only: good for echo frames)"},
    }
    r["any_feasible"] = any(o["feasible"] for o in r["options"].values())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace_check.json")
    ap.add_argument("--no-fetch", action="store_true", help="skip the pip download test")
    a = ap.parse_args()
    r = run(fetch=not a.no_fetch)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"python {r['python']}  GPU: {r['gpu']}  disk free: {r['disk_free_gb']} GB")
    for k, o in r["options"].items():
        print(f"  {'YES' if o['feasible'] else 'no '}  {k:30s} needs: {o['needs']}")
    if not r["any_feasible"]:
        print("No annotation tool can run here as configured: send workspace_check.json to MCP with the questions in annotation/README.md.")


if __name__ == "__main__":
    main()
