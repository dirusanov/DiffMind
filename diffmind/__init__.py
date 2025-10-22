from importlib import metadata


try:
    __version__ = metadata.version("diffmind")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

