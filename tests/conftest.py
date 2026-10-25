import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def no_network(monkeypatch):
    """Fail any test that tries to open a network connection."""

    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
