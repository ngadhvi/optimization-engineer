import torch
import psutil
from typing import Optional

def get_device(device: Optional[str] = None) -> str:
    """Auto-detect or validate device."""
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device

def get_system_info() -> str:
    """Get formatted system information."""
    info = ["# System Information\n"]
    
    # CPU
    info.append(f"**CPU**: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical cores")
    info.append(f"**Memory**: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # GPU
    if torch.cuda.is_available():
        info.append(f"**CUDA**: {torch.cuda.get_device_name(0)}")
        info.append(f"**CUDA Memory**: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        info.append(f"**CUDA Version**: {torch.version.cuda}")
    
    # MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info.append("**Apple Silicon**: MPS Available")
    
    info.append(f"**PyTorch**: {torch.__version__}")
    
    return "\n".join(info)
