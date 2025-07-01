"""
Agentic Kernel Framework for Kokkos-Central

This module implements agentic kernels with explicit tensor shapes following
hypergraph pattern encoding. Each kernel is represented as a Scheme-style
dictionary with tensor shapes determined by degrees of freedom and computational depth.

Optimized for cognitive synergy and future ggml customization.
"""

from .kernel_dict import AgenticKernel, TensorShape, HypergraphPattern
from .kernel_factory import KernelFactory, extract_from_module
from .cognitive_optimizer import CognitiveOptimizer
from .ggml_compatibility import GGMLAdapter

__all__ = [
    'AgenticKernel',
    'TensorShape', 
    'HypergraphPattern',
    'KernelFactory',
    'extract_from_module',
    'CognitiveOptimizer',
    'GGMLAdapter'
]