"""Built-in device memory profiles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    memory_gb: float
    description: str


DEVICES = {
    "m3pro18": DeviceProfile("m3pro18", 18.0, "Apple M3 Pro with 18 GB unified memory"),
    "m3-pro-18gb": DeviceProfile("m3-pro-18gb", 18.0, "Apple M3 Pro with 18 GB unified memory"),
    "rtx4090": DeviceProfile("rtx4090", 24.0, "NVIDIA GeForce RTX 4090"),
    "a10g": DeviceProfile("a10g", 24.0, "NVIDIA A10G"),
    "a10g": DeviceProfile("a10g", 24.0, "NVIDIA A10G"),
    "h100": DeviceProfile("h100", 80.0, "NVIDIA H100 80 GB"),
}


def get_device(name: str) -> DeviceProfile:
    try:
        return DEVICES[name.casefold()]
    except KeyError as error:
        choices = ", ".join(sorted(DEVICES))
        raise ValueError(f"Unknown device {name!r}. Choose from: {choices}.") from error
