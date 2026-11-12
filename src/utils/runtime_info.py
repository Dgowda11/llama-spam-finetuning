import torch

def get_gpu_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU available"
