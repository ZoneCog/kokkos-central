"""
Cognitive optimization for agentic kernels.

This module implements cognitive synergy optimization algorithms for agentic kernels,
focusing on improving computational efficiency through pattern recognition and
adaptive optimization strategies.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import networkx as nx

# Note: For standalone testing, we'll import relatively later
# from .kernel_dict import AgenticKernel, HypergraphPattern, TensorShape


@dataclass
class CognitiveMetrics:
    """Metrics for evaluating cognitive synergy of kernel patterns."""
    connectivity_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_score: float = 0.0
    synergy_score: float = 0.0
    adaptability_score: float = 0.0


class CognitiveOptimizer:
    """
    Optimizer for cognitive synergy in agentic kernel systems.
    
    Uses pattern recognition and graph theory to optimize kernel representations
    for improved cognitive processing and computational efficiency.
    """
    
    def __init__(self, learning_rate: float = 0.1, adaptation_threshold: float = 0.8):
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.optimization_history: List[Dict[str, Any]] = []
        self.pattern_library: Dict[str, Any] = {}
    
    def optimize_kernel(self, kernel: AgenticKernel) -> AgenticKernel:
        """
        Optimize a single agentic kernel for cognitive synergy.
        
        Args:
            kernel: The agentic kernel to optimize
            
        Returns:
            Optimized agentic kernel
        """
        # Calculate current cognitive metrics
        current_metrics = self._calculate_cognitive_metrics(kernel)
        
        # Apply optimization strategies
        optimized_kernel = self._apply_optimization_strategies(kernel, current_metrics)
        
        # Verify improvement
        new_metrics = self._calculate_cognitive_metrics(optimized_kernel)
        
        # Record optimization history
        self.optimization_history.append({
            'kernel_name': kernel.name,
            'before_metrics': current_metrics,
            'after_metrics': new_metrics,
            'improvement': new_metrics.synergy_score - current_metrics.synergy_score
        })
        
        return optimized_kernel
    
    def optimize_kernel_collection(self, kernels: List[AgenticKernel]) -> List[AgenticKernel]:
        """
        Optimize a collection of agentic kernels with inter-kernel synergy.
        
        Args:
            kernels: List of agentic kernels to optimize
            
        Returns:
            List of optimized agentic kernels
        """
        if not kernels:
            return kernels
        
        # First, optimize individual kernels
        optimized_kernels = [self.optimize_kernel(k) for k in kernels]
        
        # Then optimize for inter-kernel synergy
        optimized_kernels = self._optimize_inter_kernel_synergy(optimized_kernels)
        
        # Apply collective pattern recognition
        optimized_kernels = self._apply_collective_patterns(optimized_kernels)
        
        return optimized_kernels
    
    def _calculate_cognitive_metrics(self, kernel: AgenticKernel) -> CognitiveMetrics:
        """Calculate cognitive metrics for a kernel."""
        metrics = CognitiveMetrics()
        
        # Connectivity score based on hypergraph structure
        metrics.connectivity_score = self._calculate_connectivity_score(kernel.hypergraph)
        
        # Complexity score based on tensor shapes and computational depth
        metrics.complexity_score = self._calculate_complexity_score(kernel)
        
        # Efficiency score based on memory requirements and DOF
        metrics.efficiency_score = self._calculate_efficiency_score(kernel)
        
        # Adaptability score based on pattern flexibility
        metrics.adaptability_score = self._calculate_adaptability_score(kernel)
        
        # Overall synergy score (weighted combination)
        metrics.synergy_score = (
            0.3 * metrics.connectivity_score +
            0.2 * metrics.complexity_score +
            0.3 * metrics.efficiency_score +
            0.2 * metrics.adaptability_score
        )
        
        return metrics
    
    def _calculate_connectivity_score(self, hypergraph: HypergraphPattern) -> float:
        """Calculate connectivity score from hypergraph structure."""
        if not hypergraph.vertices or not hypergraph.hyperedges:
            return 0.0
        
        # Build graph representation for analysis
        G = nx.Graph()
        G.add_nodes_from(hypergraph.vertices)
        
        # Add edges from hyperedges (convert hyperedges to pairwise edges)
        for hyperedge in hypergraph.hyperedges:
            for i, v1 in enumerate(hyperedge):
                for v2 in hyperedge[i+1:]:
                    weight = hypergraph.edge_weights.get(hyperedge, 1.0)
                    if G.has_edge(v1, v2):
                        G[v1][v2]['weight'] += weight
                    else:
                        G.add_edge(v1, v2, weight=weight)
        
        # Calculate various connectivity metrics
        try:
            density = nx.density(G)
            if len(G.nodes()) > 1:
                avg_clustering = nx.average_clustering(G)
            else:
                avg_clustering = 0.0
            
            # Connectivity score combines density and clustering
            connectivity_score = 0.6 * density + 0.4 * avg_clustering
            
        except (nx.NetworkXError, ZeroDivisionError):
            connectivity_score = 0.0
        
        return min(1.0, connectivity_score)
    
    def _calculate_complexity_score(self, kernel: AgenticKernel) -> float:
        """Calculate complexity score based on kernel structure."""
        # Factors affecting complexity
        num_tensors = len(kernel.input_tensors) + len(kernel.output_tensors)
        total_dof = kernel.calculate_total_dof()
        computational_depth = kernel.computational_depth
        
        # Normalize complexity factors
        tensor_complexity = min(1.0, num_tensors / 10.0)  # Normalize to 10 tensors
        dof_complexity = min(1.0, np.log10(max(1, total_dof)) / 6.0)  # Log scale, normalize to 10^6
        depth_complexity = min(1.0, computational_depth / 10.0)  # Normalize to depth 10
        
        # Lower complexity is better (inverse relationship)
        complexity_score = 1.0 - (0.4 * tensor_complexity + 0.4 * dof_complexity + 0.2 * depth_complexity)
        
        return max(0.0, complexity_score)
    
    def _calculate_efficiency_score(self, kernel: AgenticKernel) -> float:
        """Calculate efficiency score based on memory and computation ratios."""
        total_dof = kernel.calculate_total_dof()
        memory_req = kernel.calculate_memory_requirements()
        
        if total_dof == 0 or memory_req == 0:
            return 0.0
        
        # Memory efficiency (bytes per DOF)
        bytes_per_dof = memory_req / total_dof
        memory_efficiency = 1.0 / (1.0 + bytes_per_dof / 4.0)  # Normalize to 4 bytes/DOF (float32)
        
        # Computational efficiency (based on parallelism pattern)
        parallelism_efficiency = {
            'data_parallel': 0.9,
            'reduction': 0.8,
            'hierarchical': 0.7,
            'serial': 0.3
        }.get(kernel.parallelism_pattern, 0.5)
        
        efficiency_score = 0.6 * memory_efficiency + 0.4 * parallelism_efficiency
        
        return min(1.0, efficiency_score)
    
    def _calculate_adaptability_score(self, kernel: AgenticKernel) -> float:
        """Calculate adaptability score based on kernel flexibility."""
        # Check for dynamic dimensions
        has_dynamic_dims = any(
            any(d <= 0 for d in tensor.dimensions) 
            for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values())
        )
        
        # Check for multiple tensor layouts
        unique_layouts = set(
            tensor.layout 
            for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values())
        )
        
        # Check for sparse tensor support
        has_sparse_tensors = any(
            tensor.sparsity_pattern is not None
            for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values())
        )
        
        # Adaptability factors
        dynamic_score = 0.4 if has_dynamic_dims else 0.0
        layout_score = min(0.3, len(unique_layouts) * 0.1)
        sparse_score = 0.3 if has_sparse_tensors else 0.0
        
        adaptability_score = dynamic_score + layout_score + sparse_score
        
        return min(1.0, adaptability_score)
    
    def _apply_optimization_strategies(self, kernel: AgenticKernel, metrics: CognitiveMetrics) -> AgenticKernel:
        """Apply optimization strategies based on current metrics."""
        optimized_kernel = AgenticKernel(
            name=kernel.name,
            input_tensors=kernel.input_tensors.copy(),
            output_tensors=kernel.output_tensors.copy(),
            hypergraph=HypergraphPattern(
                vertices=kernel.hypergraph.vertices.copy(),
                hyperedges=kernel.hypergraph.hyperedges.copy(),
                edge_weights=kernel.hypergraph.edge_weights.copy(),
                cognitive_affinity=kernel.hypergraph.cognitive_affinity
            ),
            computational_depth=kernel.computational_depth,
            parallelism_pattern=kernel.parallelism_pattern,
            memory_requirements=kernel.memory_requirements,
            ggml_compatible=kernel.ggml_compatible,
            metadata=kernel.metadata.copy()
        )
        
        # Strategy 1: Optimize hypergraph connectivity
        if metrics.connectivity_score < 0.5:
            optimized_kernel = self._optimize_hypergraph_connectivity(optimized_kernel)
        
        # Strategy 2: Reduce complexity if too high
        if metrics.complexity_score < 0.3:
            optimized_kernel = self._reduce_kernel_complexity(optimized_kernel)
        
        # Strategy 3: Improve efficiency
        if metrics.efficiency_score < 0.6:
            optimized_kernel = self._improve_kernel_efficiency(optimized_kernel)
        
        # Strategy 4: Enhance adaptability
        if metrics.adaptability_score < 0.4:
            optimized_kernel = self._enhance_kernel_adaptability(optimized_kernel)
        
        return optimized_kernel
    
    def _optimize_hypergraph_connectivity(self, kernel: AgenticKernel) -> AgenticKernel:
        """Optimize hypergraph connectivity patterns."""
        # Add more strategic connections between compatible tensors
        tensor_names = list(kernel.input_tensors.keys()) + list(kernel.output_tensors.keys())
        all_tensors = {**kernel.input_tensors, **kernel.output_tensors}
        
        # Find tensor pairs with similar characteristics
        for i, name1 in enumerate(tensor_names):
            for name2 in tensor_names[i+1:]:
                tensor1 = all_tensors[name1]
                tensor2 = all_tensors[name2]
                
                # Check for shape compatibility or similar computational depth
                if (len(tensor1.dimensions) == len(tensor2.dimensions) or
                    abs(tensor1.computational_depth - tensor2.computational_depth) <= 1):
                    
                    # Add hyperedge if not already present
                    edge = (name1, name2)
                    if edge not in kernel.hypergraph.hyperedges and (name2, name1) not in kernel.hypergraph.hyperedges:
                        kernel.hypergraph.add_edge(edge, weight=0.3)
        
        # Recalculate cognitive affinity
        kernel.hypergraph.calculate_cognitive_affinity()
        
        return kernel
    
    def _reduce_kernel_complexity(self, kernel: AgenticKernel) -> AgenticKernel:
        """Reduce kernel complexity through optimization."""
        # Strategy: Merge compatible tensors or reduce computational depth
        if kernel.computational_depth > 3:
            kernel.computational_depth = max(1, kernel.computational_depth - 1)
        
        # Update tensor computational depths
        for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values()):
            if tensor.computational_depth > kernel.computational_depth:
                tensor.computational_depth = kernel.computational_depth
        
        return kernel
    
    def _improve_kernel_efficiency(self, kernel: AgenticKernel) -> AgenticKernel:
        """Improve kernel efficiency."""
        # Strategy: Optimize parallelism pattern and memory layout
        total_dof = kernel.calculate_total_dof()
        
        # Choose better parallelism pattern based on problem size
        if total_dof > 10000 and kernel.parallelism_pattern == 'serial':
            kernel.parallelism_pattern = 'data_parallel'
        elif total_dof < 1000 and kernel.parallelism_pattern == 'hierarchical':
            kernel.parallelism_pattern = 'data_parallel'
        
        # Optimize tensor layouts for better memory access
        for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values()):
            if len(tensor.dimensions) == 2 and tensor.layout.value == 'col_major':
                # Prefer row-major for better cache locality in most cases
                tensor.layout = tensor.layout.__class__.ROW_MAJOR
        
        return kernel
    
    def _enhance_kernel_adaptability(self, kernel: AgenticKernel) -> AgenticKernel:
        """Enhance kernel adaptability for different scenarios."""
        # Strategy: Add support for dynamic dimensions where appropriate
        for tensor_name, tensor in kernel.input_tensors.items():
            if len(tensor.dimensions) > 1 and all(d > 100 for d in tensor.dimensions):
                # Make large tensors more flexible by supporting dynamic dimensions
                new_dims = tuple(-1 if d > 1000 else d for d in tensor.dimensions)
                tensor.dimensions = new_dims
                tensor.__post_init__()  # Recalculate DOF
        
        # Ensure GGML compatibility for better adaptability
        kernel.ggml_compatible = True
        
        return kernel
    
    def _optimize_inter_kernel_synergy(self, kernels: List[AgenticKernel]) -> List[AgenticKernel]:
        """Optimize synergy between multiple kernels."""
        if len(kernels) < 2:
            return kernels
        
        # Find kernels with compatible tensor shapes for fusion opportunities
        for i, kernel1 in enumerate(kernels):
            for j, kernel2 in enumerate(kernels[i+1:], i+1):
                # Check if output of kernel1 matches input of kernel2
                for out_name, out_tensor in kernel1.output_tensors.items():
                    for in_name, in_tensor in kernel2.input_tensors.items():
                        if (out_tensor.dimensions == in_tensor.dimensions and
                            out_tensor.layout == in_tensor.layout):
                            # Add inter-kernel connection in metadata
                            kernel1.metadata.setdefault('connected_kernels', []).append(kernel2.name)
                            kernel2.metadata.setdefault('connected_kernels', []).append(kernel1.name)
        
        return kernels
    
    def _apply_collective_patterns(self, kernels: List[AgenticKernel]) -> List[AgenticKernel]:
        """Apply collective optimization patterns across all kernels."""
        # Pattern 1: Standardize common tensor dimensions
        common_dimensions = self._find_common_dimensions(kernels)
        
        # Pattern 2: Optimize for similar computational depths
        target_depth = self._calculate_optimal_depth(kernels)
        
        for kernel in kernels:
            # Apply common dimension optimization
            for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values()):
                if tensor.dimensions in common_dimensions:
                    # Already optimal
                    continue
                    
            # Apply depth optimization
            if abs(kernel.computational_depth - target_depth) > 2:
                kernel.computational_depth = target_depth
        
        return kernels
    
    def _find_common_dimensions(self, kernels: List[AgenticKernel]) -> List[Tuple[int, ...]]:
        """Find commonly used tensor dimensions across kernels."""
        dimension_counts = {}
        
        for kernel in kernels:
            for tensor in list(kernel.input_tensors.values()) + list(kernel.output_tensors.values()):
                dims = tensor.dimensions
                dimension_counts[dims] = dimension_counts.get(dims, 0) + 1
        
        # Return dimensions used by at least 2 kernels
        return [dims for dims, count in dimension_counts.items() if count >= 2]
    
    def _calculate_optimal_depth(self, kernels: List[AgenticKernel]) -> int:
        """Calculate optimal computational depth for the kernel collection."""
        if not kernels:
            return 1
        
        depths = [k.computational_depth for k in kernels]
        # Use median as optimal depth to balance between extremes
        return int(np.median(depths))
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history and metrics."""
        if not self.optimization_history:
            return {'message': 'No optimizations performed yet'}
        
        improvements = [entry['improvement'] for entry in self.optimization_history]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'average_improvement': np.mean(improvements),
            'best_improvement': max(improvements),
            'total_improvement': sum(improvements),
            'kernels_optimized': [entry['kernel_name'] for entry in self.optimization_history]
        }