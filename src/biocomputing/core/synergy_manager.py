"""Manages synergistic interactions between phenomena."""

from typing import List, Dict, Any
from .base_phenomenon import BasePhenomenon

class SynergyManager:
    def __init__(self):
        self.phenomena = []
    def add_phenomenon(self, phenomenon: BasePhenomenon) -> None:
        self.phenomena.append(phenomenon)
