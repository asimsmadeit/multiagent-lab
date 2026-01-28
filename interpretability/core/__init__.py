"""Core components for InterpretabilityRunner.

This module contains helper classes that handle specific responsibilities,
extracted from the monolithic InterpretabilityRunner class.

Classes:
    DatasetBuilder: Builds and saves activation datasets for probe training
    GroundTruthDetector: Extracts deception labels from agent behavior
"""

from .dataset_builder import DatasetBuilder
from .ground_truth import GroundTruthDetector

__all__ = [
    "DatasetBuilder",
    "GroundTruthDetector",
]
