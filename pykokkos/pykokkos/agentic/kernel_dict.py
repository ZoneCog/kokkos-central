"""
Core data structures for agentic kernels using Scheme-style dictionaries.

This module defines the fundamental building blocks for representing computational
kernels as agentic entities with explicit tensor shapes and hypergraph patterns.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TensorLayout(Enum):
    """Tensor memory layout patterns for optimization."""
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    SPARSE_COO = "sparse_coo"
    SPARSE_CSR = "sparse_csr"
    HYPERGRAPH = "hypergraph"


@dataclass
class TensorShape:
    """
    Explicit tensor shape with degrees of freedom tracking.
    
    Represents tensor dimensions with support for dynamic shapes,
    sparsity patterns, and hypergraph connectivity.
    """
    dimensions: Tuple[int, ...]
    layout: TensorLayout = TensorLayout.ROW_MAJOR
    sparsity_pattern: Optional[np.ndarray] = None
    degrees_of_freedom: int = field(init=False)
    computational_depth: int = 0
    
    def __post_init__(self):
        """Calculate degrees of freedom based on shape and sparsity."""
        if self.sparsity_pattern is not None:
            self.degrees_of_freedom = np.count_nonzero(self.sparsity_pattern)
        else:
            self.degrees_of_freedom = np.prod(self.dimensions)
    
    def to_scheme_dict(self) -> Dict[str, Any]:
        """Convert to Scheme-style dictionary representation."""
        return {
            'dimensions': list(self.dimensions),
            'layout': self.layout.value,
            'degrees_of_freedom': self.degrees_of_freedom,
            'computational_depth': self.computational_depth,
            'sparsity_pattern': self.sparsity_pattern.tolist() if self.sparsity_pattern is not None else None
        }


@dataclass 
class HypergraphPattern:
    """
    Hypergraph encoding for kernel connectivity patterns.
    
    Represents the computational graph structure using hypergraph theory
    to capture multi-way dependencies and optimize for cognitive synergy.
    """
    vertices: List[str]  # Kernel components/operations
    hyperedges: List[Tuple[str, ...]] = field(default_factory=list)  # Multi-way connections
    edge_weights: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    cognitive_affinity: float = 0.0
    
    def add_edge(self, vertices: Tuple[str, ...], weight: float = 1.0):
        """Add a hyperedge connecting multiple vertices."""
        if all(v in self.vertices for v in vertices):
            self.hyperedges.append(vertices)
            self.edge_weights[vertices] = weight
    
    def calculate_cognitive_affinity(self) -> float:
        """Calculate cognitive synergy metric based on hypergraph structure."""
        if not self.hyperedges:
            return 0.0
        
        # Simple heuristic: higher affinity for more interconnected patterns
        total_connectivity = sum(len(edge) for edge in self.hyperedges)
        unique_vertices = len(set(v for edge in self.hyperedges for v in edge))
        
        if unique_vertices == 0:
            return 0.0
            
        self.cognitive_affinity = total_connectivity / unique_vertices
        return self.cognitive_affinity
    
    def to_scheme_dict(self) -> Dict[str, Any]:
        """Convert to Scheme-style dictionary representation."""
        return {
            'vertices': self.vertices,
            'hyperedges': [list(edge) for edge in self.hyperedges],
            'edge_weights': {str(k): v for k, v in self.edge_weights.items()},
            'cognitive_affinity': self.cognitive_affinity
        }


@dataclass
class AgenticKernel:
    """
    Main agentic kernel representation using Scheme-style dictionary.
    
    Encapsulates a computational kernel with explicit tensor shapes,
    hypergraph patterns, and cognitive optimization metadata.
    """
    name: str
    input_tensors: Dict[str, TensorShape]
    output_tensors: Dict[str, TensorShape]
    hypergraph: HypergraphPattern
    computational_depth: int = 0
    parallelism_pattern: str = "data_parallel"
    memory_requirements: int = 0
    ggml_compatible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total_dof(self) -> int:
        """Calculate total degrees of freedom across all tensors."""
        input_dof = sum(t.degrees_of_freedom for t in self.input_tensors.values())
        output_dof = sum(t.degrees_of_freedom for t in self.output_tensors.values())
        return input_dof + output_dof
    
    def calculate_memory_requirements(self) -> int:
        """Estimate memory requirements in bytes."""
        total_elements = 0
        for tensor in list(self.input_tensors.values()) + list(self.output_tensors.values()):
            total_elements += tensor.degrees_of_freedom
        
        # Assume 4 bytes per float32 element
        self.memory_requirements = total_elements * 4
        return self.memory_requirements
    
    def optimize_for_cognitive_synergy(self):
        """Optimize kernel representation for cognitive synergy."""
        self.hypergraph.calculate_cognitive_affinity()
        
        # Adjust computational depth based on cognitive affinity
        if self.hypergraph.cognitive_affinity > 1.5:
            self.computational_depth = max(1, self.computational_depth - 1)
        elif self.hypergraph.cognitive_affinity < 0.5:
            self.computational_depth += 1
    
    def to_scheme_dict(self) -> Dict[str, Any]:
        """Convert entire kernel to Scheme-style dictionary."""
        return {
            'name': self.name,
            'input_tensors': {k: v.to_scheme_dict() for k, v in self.input_tensors.items()},
            'output_tensors': {k: v.to_scheme_dict() for k, v in self.output_tensors.items()},
            'hypergraph': self.hypergraph.to_scheme_dict(),
            'computational_depth': self.computational_depth,
            'parallelism_pattern': self.parallelism_pattern,
            'total_degrees_of_freedom': self.calculate_total_dof(),
            'memory_requirements': self.calculate_memory_requirements(),
            'ggml_compatible': self.ggml_compatible,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_scheme_dict(cls, data: Dict[str, Any]) -> 'AgenticKernel':
        """Create AgenticKernel from Scheme-style dictionary."""
        # Convert tensor shapes
        input_tensors = {}
        for k, v in data['input_tensors'].items():
            shape = TensorShape(
                dimensions=tuple(v['dimensions']),
                layout=TensorLayout(v['layout']),
                computational_depth=v['computational_depth']
            )
            if v['sparsity_pattern'] is not None:
                shape.sparsity_pattern = np.array(v['sparsity_pattern'])
            input_tensors[k] = shape
        
        output_tensors = {}
        for k, v in data['output_tensors'].items():
            shape = TensorShape(
                dimensions=tuple(v['dimensions']),
                layout=TensorLayout(v['layout']),
                computational_depth=v['computational_depth']
            )
            if v['sparsity_pattern'] is not None:
                shape.sparsity_pattern = np.array(v['sparsity_pattern'])
            output_tensors[k] = shape
        
        # Convert hypergraph
        hypergraph_data = data['hypergraph']
        hypergraph = HypergraphPattern(
            vertices=hypergraph_data['vertices'],
            hyperedges=[tuple(edge) for edge in hypergraph_data['hyperedges']],
            cognitive_affinity=hypergraph_data['cognitive_affinity']
        )
        
        return cls(
            name=data['name'],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            hypergraph=hypergraph,
            computational_depth=data['computational_depth'],
            parallelism_pattern=data['parallelism_pattern'],
            ggml_compatible=data['ggml_compatible'],
            metadata=data.get('metadata', {})
        )