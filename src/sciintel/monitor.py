from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def system_report() -> dict:
    report = {"python": sys.version.split()[0], "platform": platform.platform(),
              "cpu_count": None, "ram_total_gb": None, "ram_available_gb": None,
              "cuda_available": False, "torch": None, "torch_cuda": None,
              "gpu_name": None, "gpu_vram_gb": None, "nvidia_smi": None}
    try:
        import psutil
        report["cpu_count"] = psutil.cpu_count(logical=True)
        report["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        report["ram_available_gb"] = round(psutil.virtual_memory().available / 1024**3, 2)
    except Exception:
        pass
    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception as exc:
        report["torch_error"] = str(exc)
    if shutil.which("nvidia-smi"):
        try:
            output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10, check=False)
            report["nvidia_smi"] = output.stdout.strip() or output.stderr.strip()
        except Exception as exc:
            report["nvidia_smi"] = str(exc)
    return report
