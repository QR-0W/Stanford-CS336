import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-scaling")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+local"
