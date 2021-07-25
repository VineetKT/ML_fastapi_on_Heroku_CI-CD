"""
pytest module for main.py
"""

from main import add


def test_main():
    assert add(5, 7) == 12
