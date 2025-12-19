"""
Demo utilities for Constitutional AI Gradio interface.

This package provides modular components for the interactive demo:
- model_manager: Model loading and state management
- evaluator_ui: Evaluation tab logic
- comparison_ui: Before/After comparison tab
- filter_ui: Safety filter tab
- pipeline_ui: Interactive pipeline demonstration
- principle_explorer: Educational principle content
- visualizations: Chart generation
- formatters: Result formatting
- examples: Test prompts and examples
"""

from .model_manager import ModelManager, ModelStatus
from .examples import get_example_prompts, get_test_suites
from .formatters import (
    format_evaluation_result,
    format_filter_result,
    format_comparison_result,
)

__all__ = [
    "ModelManager",
    "ModelStatus",
    "get_example_prompts",
    "get_test_suites",
    "format_evaluation_result",
    "format_filter_result",
    "format_comparison_result",
]
