#!/usr/bin/env python3
"""
Example script demonstrating the agentic kernel framework.

This script shows how to create, optimize, and convert agentic kernels
for various computational patterns found in the Kokkos ecosystem.
"""

import sys
import os
from pathlib import Path

# Add pykokkos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pykokkos.agentic import (
        AgenticKernel, TensorShape, HypergraphPattern, TensorLayout,
        KernelFactory, CognitiveOptimizer, GGMLAdapter
    )
    import numpy as np
    import json
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


def create_matrix_multiply_kernel():
    """Create an agentic kernel for matrix multiplication."""
    print("Creating matrix multiplication agentic kernel...")
    
    # Define input tensors
    matrix_a = TensorShape(
        dimensions=(1024, 512),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=1
    )
    
    matrix_b = TensorShape(
        dimensions=(512, 256),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=1
    )
    
    # Define output tensor
    result_matrix = TensorShape(
        dimensions=(1024, 256),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=2
    )
    
    # Create hypergraph pattern
    hypergraph = HypergraphPattern(
        vertices=['kernel', 'matrix_a', 'matrix_b', 'result']
    )
    hypergraph.add_edge(('kernel', 'matrix_a', 'matrix_b'), weight=1.0)
    hypergraph.add_edge(('kernel', 'result'), weight=1.0)
    hypergraph.add_edge(('matrix_a', 'matrix_b', 'result'), weight=0.8)
    
    # Create agentic kernel
    kernel = AgenticKernel(
        name='matrix_multiply_kernel',
        input_tensors={
            'matrix_a': matrix_a,
            'matrix_b': matrix_b
        },
        output_tensors={
            'result': result_matrix
        },
        hypergraph=hypergraph,
        computational_depth=2,
        parallelism_pattern='data_parallel',
        metadata={
            'operation_type': 'linear_algebra',
            'flops': 2 * 1024 * 512 * 256,  # 2 * M * K * N
            'memory_bound': False
        }
    )
    
    return kernel


def create_reduction_kernel():
    """Create an agentic kernel for parallel reduction."""
    print("Creating parallel reduction agentic kernel...")
    
    # Define input tensor
    input_array = TensorShape(
        dimensions=(1000000,),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=1
    )
    
    # Define output tensor (scalar result)
    result_scalar = TensorShape(
        dimensions=(1,),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=3
    )
    
    # Create hypergraph pattern
    hypergraph = HypergraphPattern(
        vertices=['kernel', 'input_array', 'result']
    )
    hypergraph.add_edge(('kernel', 'input_array'), weight=1.0)
    hypergraph.add_edge(('kernel', 'result'), weight=1.0)
    hypergraph.add_edge(('input_array', 'result'), weight=0.9)
    
    # Create agentic kernel
    kernel = AgenticKernel(
        name='parallel_reduction_kernel',
        input_tensors={'input_array': input_array},
        output_tensors={'result': result_scalar},
        hypergraph=hypergraph,
        computational_depth=3,
        parallelism_pattern='reduction',
        metadata={
            'operation_type': 'reduction',
            'reduction_op': 'sum',
            'memory_bound': True
        }
    )
    
    return kernel


def create_sparse_kernel():
    """Create an agentic kernel for sparse matrix operations."""
    print("Creating sparse matrix agentic kernel...")
    
    # Create sparsity pattern (90% sparse)
    sparsity_pattern = np.random.choice([0, 1], size=(1000, 1000), p=[0.9, 0.1])
    
    # Define sparse input tensor
    sparse_matrix = TensorShape(
        dimensions=(1000, 1000),
        layout=TensorLayout.SPARSE_CSR,
        sparsity_pattern=sparsity_pattern,
        computational_depth=1
    )
    
    # Define dense vector
    dense_vector = TensorShape(
        dimensions=(1000,),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=1
    )
    
    # Define output vector
    result_vector = TensorShape(
        dimensions=(1000,),
        layout=TensorLayout.ROW_MAJOR,
        computational_depth=2
    )
    
    # Create hypergraph pattern
    hypergraph = HypergraphPattern(
        vertices=['kernel', 'sparse_matrix', 'dense_vector', 'result']
    )
    hypergraph.add_edge(('kernel', 'sparse_matrix', 'dense_vector'), weight=1.0)
    hypergraph.add_edge(('kernel', 'result'), weight=1.0)
    hypergraph.add_edge(('sparse_matrix', 'dense_vector', 'result'), weight=0.7)
    
    # Create agentic kernel
    kernel = AgenticKernel(
        name='sparse_matvec_kernel',
        input_tensors={
            'sparse_matrix': sparse_matrix,
            'dense_vector': dense_vector
        },
        output_tensors={'result': result_vector},
        hypergraph=hypergraph,
        computational_depth=2,
        parallelism_pattern='data_parallel',
        metadata={
            'operation_type': 'sparse_linear_algebra',
            'sparsity_ratio': 0.9,
            'memory_bound': True
        }
    )
    
    return kernel


def demonstrate_kernel_extraction():
    """Demonstrate extraction of kernels from existing code."""
    print("\nDemonstrating kernel extraction from repository modules...")
    
    factory = KernelFactory()
    
    # Try to extract kernels from existing PyKokkos files
    pykokkos_dir = Path(__file__).parent.parent / "pykokkos"
    
    if pykokkos_dir.exists():
        print(f"Scanning PyKokkos directory: {pykokkos_dir}")
        kernels = factory.extract_from_directory(str(pykokkos_dir), recursive=True)
        print(f"Found {len(kernels)} agentic kernels in repository")
        
        for i, kernel in enumerate(kernels[:3]):  # Show first 3
            print(f"  {i+1}. {kernel.name} - {kernel.calculate_total_dof()} DOF")
    else:
        print("PyKokkos directory not found, skipping extraction demo")


def demonstrate_optimization():
    """Demonstrate cognitive optimization of kernels."""
    print("\nDemonstrating cognitive optimization...")
    
    # Create test kernels
    kernels = [
        create_matrix_multiply_kernel(),
        create_reduction_kernel(),
        create_sparse_kernel()
    ]
    
    # Create optimizer
    optimizer = CognitiveOptimizer()
    
    print(f"Optimizing {len(kernels)} kernels...")
    optimized_kernels = optimizer.optimize_kernel_collection(kernels)
    
    # Show optimization results
    summary = optimizer.get_optimization_summary()
    print(f"Optimization summary: {summary}")
    
    for i, (original, optimized) in enumerate(zip(kernels, optimized_kernels)):
        print(f"\nKernel {i+1}: {original.name}")
        print(f"  Original cognitive affinity: {original.hypergraph.cognitive_affinity:.3f}")
        print(f"  Optimized cognitive affinity: {optimized.hypergraph.cognitive_affinity:.3f}")
        print(f"  Original computational depth: {original.computational_depth}")
        print(f"  Optimized computational depth: {optimized.computational_depth}")
    
    return optimized_kernels


def demonstrate_ggml_conversion(kernels):
    """Demonstrate GGML conversion and compatibility."""
    print("\nDemonstrating GGML conversion...")
    
    adapter = GGMLAdapter()
    
    for kernel in kernels:
        print(f"\nProcessing kernel: {kernel.name}")
        
        # Validate compatibility
        validation = adapter.validate_ggml_compatibility(kernel)
        print(f"  GGML compatibility score: {validation['compatibility_score']:.3f}")
        
        if validation['issues']:
            print(f"  Issues: {len(validation['issues'])}")
            for issue in validation['issues'][:2]:  # Show first 2
                print(f"    - {issue}")
        
        # Optimize for GGML
        ggml_optimized = adapter.optimize_for_ggml(kernel)
        
        # Convert to GGML
        ggml_graph = adapter.convert_kernel_to_ggml(ggml_optimized)
        print(f"  Converted to GGML: {len(ggml_graph.tensors)} tensors, {len(ggml_graph.operators)} operators")
        
        # Generate C code sample
        c_code = adapter.generate_ggml_c_code(ggml_graph)
        print(f"  Generated {len(c_code)} characters of C code")


def demonstrate_scheme_dict_representation():
    """Demonstrate Scheme-style dictionary representation."""
    print("\nDemonstrating Scheme-style dictionary representation...")
    
    kernel = create_matrix_multiply_kernel()
    kernel.optimize_for_cognitive_synergy()
    
    # Convert to Scheme-style dictionary
    scheme_dict = kernel.to_scheme_dict()
    
    print(f"Kernel '{kernel.name}' as Scheme-style dictionary:")
    print(f"  Total keys: {len(scheme_dict)}")
    print(f"  Input tensors: {len(scheme_dict['input_tensors'])}")
    print(f"  Output tensors: {len(scheme_dict['output_tensors'])}")
    print(f"  Hypergraph vertices: {len(scheme_dict['hypergraph']['vertices'])}")
    print(f"  Hypergraph edges: {len(scheme_dict['hypergraph']['hyperedges'])}")
    print(f"  Total DOF: {scheme_dict['total_degrees_of_freedom']}")
    print(f"  Memory requirements: {scheme_dict['memory_requirements']} bytes")
    print(f"  GGML compatible: {scheme_dict['ggml_compatible']}")
    
    # Save as JSON for inspection
    output_file = "/tmp/agentic_kernel_example.json"
    with open(output_file, 'w') as f:
        json.dump(scheme_dict, f, indent=2, default=str)
    print(f"  Saved full representation to: {output_file}")
    
    # Test round-trip conversion
    reconstructed = AgenticKernel.from_scheme_dict(scheme_dict)
    print(f"  Round-trip conversion successful: {reconstructed.name == kernel.name}")


def main():
    """Main demonstration function."""
    print("=== Agentic Kernel Framework Demonstration ===")
    print("This demo showcases the key features of the agentic kernel system:")
    print("- Scheme-style dictionary representation")
    print("- Hypergraph pattern encoding")
    print("- Cognitive synergy optimization")
    print("- GGML compatibility layer")
    print("- Kernel extraction from existing code")
    print()
    
    try:
        # Create example kernels
        print("1. Creating example agentic kernels...")
        kernels = [
            create_matrix_multiply_kernel(),
            create_reduction_kernel(),
            create_sparse_kernel()
        ]
        print(f"Created {len(kernels)} example kernels\n")
        
        # Demonstrate Scheme-style representation
        demonstrate_scheme_dict_representation()
        
        # Demonstrate kernel extraction
        demonstrate_kernel_extraction()
        
        # Demonstrate optimization
        optimized_kernels = demonstrate_optimization()
        
        # Demonstrate GGML conversion
        demonstrate_ggml_conversion(optimized_kernels)
        
        print("\n=== Demonstration Complete ===")
        print("The agentic kernel framework successfully:")
        print("✓ Created kernels with explicit tensor shapes")
        print("✓ Applied hypergraph pattern encoding")
        print("✓ Optimized for cognitive synergy")
        print("✓ Generated GGML-compatible representations")
        print("✓ Provided Scheme-style dictionary interface")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())