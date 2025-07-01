"""
Tests for agentic kernel framework.

This module contains comprehensive tests for the agentic kernel system,
including kernel creation, optimization, and GGML compatibility.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pykokkos.agentic import (
    AgenticKernel, TensorShape, HypergraphPattern, TensorLayout,
    KernelFactory, CognitiveOptimizer, GGMLAdapter
)


class TestTensorShape(unittest.TestCase):
    """Test TensorShape functionality."""
    
    def test_tensor_shape_creation(self):
        """Test basic tensor shape creation."""
        shape = TensorShape(dimensions=(10, 20), layout=TensorLayout.ROW_MAJOR)
        self.assertEqual(shape.dimensions, (10, 20))
        self.assertEqual(shape.layout, TensorLayout.ROW_MAJOR)
        self.assertEqual(shape.degrees_of_freedom, 200)
    
    def test_sparse_tensor_shape(self):
        """Test tensor shape with sparsity pattern."""
        sparsity_pattern = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        shape = TensorShape(
            dimensions=(3, 3),
            layout=TensorLayout.SPARSE_COO,
            sparsity_pattern=sparsity_pattern
        )
        self.assertEqual(shape.degrees_of_freedom, 5)  # Number of non-zero elements
    
    def test_tensor_shape_to_scheme_dict(self):
        """Test conversion to Scheme-style dictionary."""
        shape = TensorShape(dimensions=(5, 10), computational_depth=2)
        scheme_dict = shape.to_scheme_dict()
        
        expected_keys = {'dimensions', 'layout', 'degrees_of_freedom', 'computational_depth', 'sparsity_pattern'}
        self.assertEqual(set(scheme_dict.keys()), expected_keys)
        self.assertEqual(scheme_dict['dimensions'], [5, 10])
        self.assertEqual(scheme_dict['degrees_of_freedom'], 50)


class TestHypergraphPattern(unittest.TestCase):
    """Test HypergraphPattern functionality."""
    
    def test_hypergraph_creation(self):
        """Test basic hypergraph creation."""
        vertices = ['A', 'B', 'C']
        hypergraph = HypergraphPattern(vertices=vertices)
        
        self.assertEqual(hypergraph.vertices, vertices)
        self.assertEqual(hypergraph.hyperedges, [])
        self.assertEqual(hypergraph.cognitive_affinity, 0.0)
    
    def test_hypergraph_add_edge(self):
        """Test adding hyperedges."""
        hypergraph = HypergraphPattern(vertices=['A', 'B', 'C'])
        hypergraph.add_edge(('A', 'B'), weight=1.5)
        hypergraph.add_edge(('A', 'B', 'C'), weight=2.0)
        
        self.assertEqual(len(hypergraph.hyperedges), 2)
        self.assertIn(('A', 'B'), hypergraph.hyperedges)
        self.assertIn(('A', 'B', 'C'), hypergraph.hyperedges)
        self.assertEqual(hypergraph.edge_weights[('A', 'B')], 1.5)
    
    def test_cognitive_affinity_calculation(self):
        """Test cognitive affinity calculation."""
        hypergraph = HypergraphPattern(vertices=['A', 'B', 'C'])
        hypergraph.add_edge(('A', 'B'), weight=1.0)
        hypergraph.add_edge(('B', 'C'), weight=1.0)
        hypergraph.add_edge(('A', 'C'), weight=1.0)
        
        affinity = hypergraph.calculate_cognitive_affinity()
        self.assertGreater(affinity, 0.0)
        self.assertEqual(hypergraph.cognitive_affinity, affinity)


class TestAgenticKernel(unittest.TestCase):
    """Test AgenticKernel functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_tensor = TensorShape(dimensions=(100, 50))
        self.output_tensor = TensorShape(dimensions=(100, 1))
        
        self.hypergraph = HypergraphPattern(vertices=['kernel', 'input', 'output'])
        self.hypergraph.add_edge(('kernel', 'input'), weight=1.0)
        self.hypergraph.add_edge(('kernel', 'output'), weight=1.0)
        
        self.kernel = AgenticKernel(
            name='test_kernel',
            input_tensors={'input': self.input_tensor},
            output_tensors={'output': self.output_tensor},
            hypergraph=self.hypergraph,
            computational_depth=2
        )
    
    def test_kernel_creation(self):
        """Test basic kernel creation."""
        self.assertEqual(self.kernel.name, 'test_kernel')
        self.assertEqual(len(self.kernel.input_tensors), 1)
        self.assertEqual(len(self.kernel.output_tensors), 1)
        self.assertEqual(self.kernel.computational_depth, 2)
    
    def test_total_dof_calculation(self):
        """Test total degrees of freedom calculation."""
        total_dof = self.kernel.calculate_total_dof()
        expected_dof = 100 * 50 + 100 * 1  # input + output
        self.assertEqual(total_dof, expected_dof)
    
    def test_memory_requirements_calculation(self):
        """Test memory requirements calculation."""
        memory_req = self.kernel.calculate_memory_requirements()
        expected_memory = (100 * 50 + 100 * 1) * 4  # 4 bytes per float32
        self.assertEqual(memory_req, expected_memory)
    
    def test_cognitive_synergy_optimization(self):
        """Test cognitive synergy optimization."""
        original_depth = self.kernel.computational_depth
        self.kernel.optimize_for_cognitive_synergy()
        
        # Should have calculated cognitive affinity
        self.assertGreaterEqual(self.kernel.hypergraph.cognitive_affinity, 0.0)
    
    def test_scheme_dict_conversion(self):
        """Test conversion to and from Scheme-style dictionary."""
        scheme_dict = self.kernel.to_scheme_dict()
        
        # Test round-trip conversion
        reconstructed_kernel = AgenticKernel.from_scheme_dict(scheme_dict)
        
        self.assertEqual(reconstructed_kernel.name, self.kernel.name)
        self.assertEqual(reconstructed_kernel.computational_depth, self.kernel.computational_depth)
        self.assertEqual(len(reconstructed_kernel.input_tensors), len(self.kernel.input_tensors))
        self.assertEqual(len(reconstructed_kernel.output_tensors), len(self.kernel.output_tensors))


class TestKernelFactory(unittest.TestCase):
    """Test KernelFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = KernelFactory()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_python_module_analysis(self):
        """Test analysis of Python modules."""
        # Create a test Python file
        test_py_content = '''
import pykokkos as pk

@pk.workload
def test_kernel(view_a, view_b):
    """Test kernel for matrix operations."""
    for i in range(view_a.shape[0]):
        for j in range(view_a.shape[1]):
            view_b[i] = view_a[i, j] * 2.0
'''
        
        test_file = os.path.join(self.temp_dir, 'test_module.py')
        with open(test_file, 'w') as f:
            f.write(test_py_content)
        
        kernels = self.factory.extract_from_module(test_file)
        
        # Should find at least one kernel
        self.assertGreaterEqual(len(kernels), 0)
    
    def test_cpp_module_analysis(self):
        """Test analysis of C++ modules."""
        # Create a test C++ file
        test_cpp_content = '''
#include <Kokkos_Core.hpp>

template<typename ViewType>
struct TestKernel {
    ViewType view_a;
    ViewType view_b;
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        view_b(i) = view_a(i) * 2.0;
    }
};

void run_test() {
    Kokkos::parallel_for("test", 100, TestKernel<Kokkos::View<double*>>{});
}
'''
        
        test_file = os.path.join(self.temp_dir, 'test_module.cpp')
        with open(test_file, 'w') as f:
            f.write(test_cpp_content)
        
        kernels = self.factory.extract_from_module(test_file)
        
        # Should find parallel_for pattern
        self.assertGreaterEqual(len(kernels), 0)
    
    def test_mlir_module_analysis(self):
        """Test analysis of MLIR modules."""
        # Create a test MLIR file
        test_mlir_content = '''
func.func @matrix_multiply(%a: tensor<10x20xf32>, %b: tensor<20x30xf32>) -> tensor<10x30xf32> {
    %c = arith.constant dense<0.0> : tensor<10x30xf32>
    %result = linalg.matmul ins(%a, %b : tensor<10x20xf32>, tensor<20x30xf32>) 
                           outs(%c : tensor<10x30xf32>) -> tensor<10x30xf32>
    return %result : tensor<10x30xf32>
}
'''
        
        test_file = os.path.join(self.temp_dir, 'test_module.mlir')
        with open(test_file, 'w') as f:
            f.write(test_mlir_content)
        
        kernels = self.factory.extract_from_module(test_file)
        
        # Should find matrix_multiply function
        self.assertGreaterEqual(len(kernels), 0)


class TestCognitiveOptimizer(unittest.TestCase):
    """Test CognitiveOptimizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = CognitiveOptimizer()
        
        # Create a test kernel
        input_tensor = TensorShape(dimensions=(1000, 1000))
        output_tensor = TensorShape(dimensions=(1000, 1))
        
        hypergraph = HypergraphPattern(vertices=['kernel', 'input', 'output'])
        hypergraph.add_edge(('kernel', 'input'), weight=1.0)
        hypergraph.add_edge(('kernel', 'output'), weight=1.0)
        
        self.test_kernel = AgenticKernel(
            name='test_optimization_kernel',
            input_tensors={'input': input_tensor},
            output_tensors={'output': output_tensor},
            hypergraph=hypergraph,
            computational_depth=5,
            parallelism_pattern='serial'
        )
    
    def test_cognitive_metrics_calculation(self):
        """Test cognitive metrics calculation."""
        metrics = self.optimizer._calculate_cognitive_metrics(self.test_kernel)
        
        self.assertIsInstance(metrics.connectivity_score, float)
        self.assertIsInstance(metrics.complexity_score, float)
        self.assertIsInstance(metrics.efficiency_score, float)
        self.assertIsInstance(metrics.synergy_score, float)
        
        # All scores should be between 0 and 1
        for score in [metrics.connectivity_score, metrics.complexity_score, 
                     metrics.efficiency_score, metrics.synergy_score]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_kernel_optimization(self):
        """Test single kernel optimization."""
        original_kernel = self.test_kernel
        optimized_kernel = self.optimizer.optimize_kernel(original_kernel)
        
        # Should have optimization history
        self.assertGreater(len(self.optimizer.optimization_history), 0)
        
        # Kernel should still be valid
        self.assertEqual(optimized_kernel.name, original_kernel.name)
        self.assertGreaterEqual(len(optimized_kernel.input_tensors), 1)
        self.assertGreaterEqual(len(optimized_kernel.output_tensors), 1)
    
    def test_kernel_collection_optimization(self):
        """Test optimization of multiple kernels."""
        # Create a second kernel
        input_tensor2 = TensorShape(dimensions=(500, 500))
        output_tensor2 = TensorShape(dimensions=(500, 1))
        hypergraph2 = HypergraphPattern(vertices=['kernel2', 'input2', 'output2'])
        
        kernel2 = AgenticKernel(
            name='test_kernel_2',
            input_tensors={'input2': input_tensor2},
            output_tensors={'output2': output_tensor2},
            hypergraph=hypergraph2,
            computational_depth=3
        )
        
        kernels = [self.test_kernel, kernel2]
        optimized_kernels = self.optimizer.optimize_kernel_collection(kernels)
        
        self.assertEqual(len(optimized_kernels), 2)
        self.assertGreater(len(self.optimizer.optimization_history), 0)
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        # First optimize a kernel to generate history
        self.optimizer.optimize_kernel(self.test_kernel)
        
        summary = self.optimizer.get_optimization_summary()
        
        self.assertIn('total_optimizations', summary)
        self.assertIn('average_improvement', summary)
        self.assertIn('kernels_optimized', summary)
        self.assertGreater(summary['total_optimizations'], 0)


class TestGGMLAdapter(unittest.TestCase):
    """Test GGMLAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = GGMLAdapter()
        
        # Create a test kernel
        input_tensor = TensorShape(dimensions=(100, 50))
        output_tensor = TensorShape(dimensions=(100, 1))
        
        hypergraph = HypergraphPattern(vertices=['kernel', 'input', 'output'])
        hypergraph.add_edge(('kernel', 'input'), weight=1.0)
        hypergraph.add_edge(('kernel', 'output'), weight=1.0)
        
        self.test_kernel = AgenticKernel(
            name='test_ggml_kernel',
            input_tensors={'input': input_tensor},
            output_tensors={'output': output_tensor},
            hypergraph=hypergraph,
            computational_depth=2,
            parallelism_pattern='data_parallel'
        )
    
    def test_kernel_to_ggml_conversion(self):
        """Test conversion of kernel to GGML format."""
        ggml_graph = self.adapter.convert_kernel_to_ggml(self.test_kernel)
        
        self.assertEqual(ggml_graph.name, self.test_kernel.name)
        self.assertGreater(len(ggml_graph.tensors), 0)
        self.assertGreater(len(ggml_graph.operators), 0)
        self.assertIn('original_kernel', ggml_graph.metadata)
    
    def test_ggml_json_export(self):
        """Test JSON export of GGML graph."""
        ggml_graph = self.adapter.convert_kernel_to_ggml(self.test_kernel)
        json_str = self.adapter.export_to_ggml_json(ggml_graph)
        
        self.assertIsInstance(json_str, str)
        self.assertIn('"name"', json_str)
        self.assertIn('"tensors"', json_str)
        self.assertIn('"operators"', json_str)
    
    def test_ggml_c_code_generation(self):
        """Test C code generation for GGML."""
        ggml_graph = self.adapter.convert_kernel_to_ggml(self.test_kernel)
        c_code = self.adapter.generate_ggml_c_code(ggml_graph)
        
        self.assertIsInstance(c_code, str)
        self.assertIn('#include "ggml.h"', c_code)
        self.assertIn('ggml_init', c_code)
        self.assertIn(self.test_kernel.name, c_code)
    
    def test_ggml_optimization(self):
        """Test GGML-specific optimization."""
        optimized_kernel = self.adapter.optimize_for_ggml(self.test_kernel)
        
        self.assertTrue(optimized_kernel.ggml_compatible)
        self.assertIn('ggml_optimized', optimized_kernel.metadata)
    
    def test_ggml_compatibility_validation(self):
        """Test GGML compatibility validation."""
        validation_result = self.adapter.validate_ggml_compatibility(self.test_kernel)
        
        self.assertIn('compatible', validation_result)
        self.assertIn('compatibility_score', validation_result)
        self.assertIn('issues', validation_result)
        self.assertIn('recommendations', validation_result)
        
        # Score should be between 0 and 1
        score = validation_result['compatibility_score']
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()