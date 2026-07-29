"""Device and mixed-precision resolution.

The target machine is a single 8 GB RTX 5060 Laptop (Blackwell, sm_120), which
supports bf16 natively -- so bf16 autocast is the default and needs no GradScaler.
fp16 remains available for an ablation, and pulls in a scaler automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    amp_dtype: torch.dtype | None  # None = autocast disabled
    use_scaler: bool  # only fp16 needs gradient scaling

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def describe(self) -> str:
        if not self.is_cuda:
            return f"device={self.device} amp=off"
        props = torch.cuda.get_device_properties(self.device)
        gb = props.total_memory / 1024**3
        amp = "off" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")
        return (
            f"device={self.device} ({props.name}, {gb:.1f} GiB, "
            f"sm_{props.major}{props.minor}) amp={amp}"
        )


def resolve_device(spec: str = "auto", amp: str = "bf16") -> DeviceInfo:
    """Pick a device and an autocast dtype, degrading safely when unsupported."""
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    amp = amp.lower()
    if amp == "off" or device.type == "cpu":
        return DeviceInfo(device, None, False)
    if amp == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            return DeviceInfo(device, torch.float16, True)  # fall back rather than crash
        return DeviceInfo(device, torch.bfloat16, False)
    if amp == "fp16":
        return DeviceInfo(device, torch.float16, True)
    raise ValueError(f"amp must be one of bf16|fp16|off, got {amp!r}")


def memory_summary(device: torch.device) -> dict[str, float]:
    """Peak memory in MiB -- worth logging when tuning batch size against 8 GB."""
    if device.type != "cuda":
        return {}
    return {
        "mem/alloc_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "mem/reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
