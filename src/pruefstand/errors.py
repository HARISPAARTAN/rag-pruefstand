"""Error types that end a CLI command with exit code 1."""


class PruefstandError(RuntimeError):
    """Base class: replay miss, cache miss, budget stop or a failed data check."""
