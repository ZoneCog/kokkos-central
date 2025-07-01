"""
Kernel factory for extracting agentic kernels from repository modules.

This module provides functionality to analyze existing Kokkos, PyKokkos, and MLIR
code and automatically generate agentic kernel representations with proper tensor
shapes and hypergraph patterns.
"""

import ast
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .kernel_dict import AgenticKernel, TensorShape, HypergraphPattern, TensorLayout
from ..core import cppast
from ..core.visitors import PyKokkosParseError


class ModuleAnalyzer:
    """Analyzes different types of repository modules to extract kernel information."""
    
    def __init__(self):
        self.supported_extensions = {'.py', '.cpp', '.hpp', '.h', '.mlir'}
    
    def analyze_python_module(self, filepath: str) -> Dict[str, Any]:
        """Analyze Python/PyKokkos module for kernel patterns."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {}
        
        kernels = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                kernel_info = self._extract_python_kernel_info(node, content)
                if kernel_info:
                    kernels.append(kernel_info)
        
        return {'type': 'python', 'kernels': kernels, 'filepath': filepath}
    
    def analyze_cpp_module(self, filepath: str) -> Dict[str, Any]:
        """Analyze C++/Kokkos module for kernel patterns."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Look for Kokkos patterns
        kernel_patterns = [
            r'parallel_for\s*\([^)]+\)',
            r'parallel_reduce\s*\([^)]+\)',
            r'parallel_scan\s*\([^)]+\)',
            r'GraphNodeKernelImpl\s*<[^>]+>',
            r'template\s*<[^>]*>\s*struct\s+\w+Kernel'
        ]
        
        kernels = []
        for pattern in kernel_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                kernel_info = self._extract_cpp_kernel_info(match, content, filepath)
                if kernel_info:
                    kernels.append(kernel_info)
        
        return {'type': 'cpp', 'kernels': kernels, 'filepath': filepath}
    
    def analyze_mlir_module(self, filepath: str) -> Dict[str, Any]:
        """Analyze MLIR module for tensor operations."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract tensor operations and shapes
        tensor_patterns = [
            r'tensor<([^>]+)>',
            r'memref<([^>]+)>',
            r'sparse_tensor\.encoding<([^>]+)>',
            r'func\.func\s+@(\w+)\s*\([^)]*\)'
        ]
        
        kernels = []
        tensor_ops = []
        
        for pattern in tensor_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if 'func.func' in pattern:
                    kernel_info = self._extract_mlir_function_info(match, content)
                    if kernel_info:
                        kernels.append(kernel_info)
                else:
                    tensor_info = self._extract_tensor_shape_info(match.group(1))
                    if tensor_info:
                        tensor_ops.append(tensor_info)
        
        return {'type': 'mlir', 'kernels': kernels, 'tensors': tensor_ops, 'filepath': filepath}
    
    def _extract_python_kernel_info(self, node: ast.FunctionDef, content: str) -> Optional[Dict[str, Any]]:
        """Extract kernel information from Python function."""
        # Look for PyKokkos decorators or patterns
        has_kokkos_decorator = any(
            hasattr(d, 'id') and 'kokkos' in d.id.lower() 
            for d in node.decorator_list 
            if hasattr(d, 'id')
        )
        
        if not has_kokkos_decorator and 'kokkos' not in node.name.lower():
            return None
        
        # Analyze function signature for tensor parameters
        input_tensors = {}
        output_tensors = {}
        
        for arg in node.args.args:
            if 'view' in arg.arg.lower() or 'tensor' in arg.arg.lower():
                # Estimate tensor shape from context
                shape = self._estimate_tensor_shape_from_name(arg.arg)
                input_tensors[arg.arg] = shape
        
        # Look for return statements to infer output tensors
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                output_tensors['result'] = self._estimate_tensor_shape_from_context(stmt, content)
        
        return {
            'name': node.name,
            'input_tensors': input_tensors,
            'output_tensors': output_tensors,
            'line_number': node.lineno,
            'parallelism_type': self._detect_parallelism_type(content, node.lineno)
        }
    
    def _extract_cpp_kernel_info(self, match: re.Match, content: str, filepath: str) -> Optional[Dict[str, Any]]:
        """Extract kernel information from C++ pattern match."""
        matched_text = match.group(0)
        
        # Determine kernel type
        if 'parallel_for' in matched_text:
            kernel_type = 'parallel_for'
        elif 'parallel_reduce' in matched_text:
            kernel_type = 'parallel_reduce'
        elif 'parallel_scan' in matched_text:
            kernel_type = 'parallel_scan'
        elif 'GraphNodeKernelImpl' in matched_text:
            kernel_type = 'graph_node'
        else:
            kernel_type = 'unknown'
        
        # Extract template parameters and estimate tensor shapes
        template_params = self._extract_template_params(matched_text)
        
        return {
            'name': f"{kernel_type}_{hash(matched_text) % 10000}",
            'type': kernel_type,
            'template_params': template_params,
            'location': match.start(),
            'filepath': filepath
        }
    
    def _extract_mlir_function_info(self, match: re.Match, content: str) -> Optional[Dict[str, Any]]:
        """Extract function information from MLIR."""
        func_name = match.group(1)
        
        # Find the full function definition
        start_pos = match.start()
        lines = content[:start_pos].count('\n')
        
        # Look for tensor types in function signature
        func_line_start = content.rfind('\n', 0, start_pos) + 1
        func_line_end = content.find('\n', match.end())
        func_signature = content[func_line_start:func_line_end]
        
        input_tensors = {}
        output_tensors = {}
        
        # Extract tensor shapes from signature
        tensor_matches = re.finditer(r'tensor<([^>]+)>', func_signature)
        tensor_idx = 0
        for tensor_match in tensor_matches:
            shape_info = self._extract_tensor_shape_info(tensor_match.group(1))
            if '>' in func_signature[tensor_match.end():tensor_match.end()+10]:
                output_tensors[f'output_{tensor_idx}'] = shape_info
            else:
                input_tensors[f'input_{tensor_idx}'] = shape_info
            tensor_idx += 1
        
        return {
            'name': func_name,
            'input_tensors': input_tensors,
            'output_tensors': output_tensors,
            'line_number': lines + 1
        }
    
    def _extract_tensor_shape_info(self, shape_str: str) -> Dict[str, Any]:
        """Extract tensor shape information from string representation."""
        # Parse shape like "10x20xf32" or "4x8xf64, #sparse_tensor.encoding"
        parts = shape_str.split(',')
        shape_part = parts[0].strip()
        
        # Extract dimensions and data type
        if 'x' in shape_part:
            tokens = shape_part.split('x')
            dimensions = []
            dtype = 'f32'  # default
            
            for token in tokens:
                if token.isdigit():
                    dimensions.append(int(token))
                elif 'f' in token:  # data type
                    dtype = token
                elif '?' in token:  # dynamic dimension
                    dimensions.append(-1)
            
            # Detect sparsity
            layout = TensorLayout.ROW_MAJOR
            if len(parts) > 1 and 'sparse' in parts[1]:
                if 'compressed' in parts[1]:
                    layout = TensorLayout.SPARSE_CSR
                else:
                    layout = TensorLayout.SPARSE_COO
        else:
            # Single dimension or special case
            dimensions = [int(shape_part)] if shape_part.isdigit() else [1]
            dtype = 'f32'
            layout = TensorLayout.ROW_MAJOR
        
        return {
            'dimensions': tuple(dimensions),
            'layout': layout,
            'dtype': dtype
        }
    
    def _estimate_tensor_shape_from_name(self, name: str) -> Dict[str, Any]:
        """Estimate tensor shape from variable name."""
        # Simple heuristics based on naming conventions
        if 'matrix' in name.lower() or 'mat' in name.lower():
            return {'dimensions': (100, 100), 'layout': TensorLayout.ROW_MAJOR, 'dtype': 'f32'}
        elif 'vector' in name.lower() or 'vec' in name.lower():
            return {'dimensions': (100,), 'layout': TensorLayout.ROW_MAJOR, 'dtype': 'f32'}
        elif 'tensor' in name.lower():
            return {'dimensions': (10, 10, 10), 'layout': TensorLayout.ROW_MAJOR, 'dtype': 'f32'}
        else:
            return {'dimensions': (1,), 'layout': TensorLayout.ROW_MAJOR, 'dtype': 'f32'}
    
    def _estimate_tensor_shape_from_context(self, node: ast.Return, content: str) -> Dict[str, Any]:
        """Estimate output tensor shape from return statement context."""
        # Default shape for unknown returns
        return {'dimensions': (1,), 'layout': TensorLayout.ROW_MAJOR, 'dtype': 'f32'}
    
    def _detect_parallelism_type(self, content: str, line_number: int) -> str:
        """Detect the type of parallelism used in the kernel."""
        # Look for parallelism keywords around the function
        context = content.split('\n')[max(0, line_number-5):line_number+5]
        context_str = '\n'.join(context).lower()
        
        if 'parallel_for' in context_str or 'prange' in context_str:
            return 'data_parallel'
        elif 'reduction' in context_str or 'reduce' in context_str:
            return 'reduction'
        elif 'hierarchical' in context_str or 'team' in context_str:
            return 'hierarchical'
        else:
            return 'serial'
    
    def _extract_template_params(self, text: str) -> List[str]:
        """Extract template parameters from C++ template syntax."""
        template_match = re.search(r'<([^>]+)>', text)
        if template_match:
            params = template_match.group(1).split(',')
            return [p.strip() for p in params]
        return []


class KernelFactory:
    """Factory for creating agentic kernels from repository modules."""
    
    def __init__(self):
        self.analyzer = ModuleAnalyzer()
        self.kernel_cache: Dict[str, AgenticKernel] = {}
    
    def extract_from_module(self, filepath: str) -> List[AgenticKernel]:
        """Extract agentic kernels from a module file."""
        if filepath in self.kernel_cache:
            return [self.kernel_cache[filepath]]
        
        ext = Path(filepath).suffix
        if ext not in self.analyzer.supported_extensions:
            return []
        
        # Analyze the module
        if ext == '.py':
            analysis = self.analyzer.analyze_python_module(filepath)
        elif ext in {'.cpp', '.hpp', '.h'}:
            analysis = self.analyzer.analyze_cpp_module(filepath)
        elif ext == '.mlir':
            analysis = self.analyzer.analyze_mlir_module(filepath)
        else:
            return []
        
        # Convert analysis to agentic kernels
        kernels = []
        for kernel_data in analysis.get('kernels', []):
            agentic_kernel = self._create_agentic_kernel(kernel_data, analysis)
            if agentic_kernel:
                kernels.append(agentic_kernel)
        
        return kernels
    
    def extract_from_directory(self, directory: str, recursive: bool = True) -> List[AgenticKernel]:
        """Extract agentic kernels from all supported files in a directory."""
        kernels = []
        
        directory_path = Path(directory)
        if recursive:
            files = directory_path.rglob('*')
        else:
            files = directory_path.glob('*')
        
        for filepath in files:
            if filepath.is_file() and filepath.suffix in self.analyzer.supported_extensions:
                module_kernels = self.extract_from_module(str(filepath))
                kernels.extend(module_kernels)
        
        return kernels
    
    def _create_agentic_kernel(self, kernel_data: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[AgenticKernel]:
        """Create an AgenticKernel from analysis data."""
        try:
            # Create tensor shapes
            input_tensors = {}
            for name, shape_data in kernel_data.get('input_tensors', {}).items():
                tensor_shape = TensorShape(
                    dimensions=shape_data['dimensions'],
                    layout=shape_data.get('layout', TensorLayout.ROW_MAJOR),
                    computational_depth=self._estimate_computational_depth(shape_data['dimensions'])
                )
                input_tensors[name] = tensor_shape
            
            output_tensors = {}
            for name, shape_data in kernel_data.get('output_tensors', {}).items():
                tensor_shape = TensorShape(
                    dimensions=shape_data['dimensions'],
                    layout=shape_data.get('layout', TensorLayout.ROW_MAJOR),
                    computational_depth=self._estimate_computational_depth(shape_data['dimensions'])
                )
                output_tensors[name] = tensor_shape
            
            # Create hypergraph pattern
            hypergraph = self._create_hypergraph_pattern(kernel_data, input_tensors, output_tensors)
            
            # Determine computational depth
            max_input_depth = max((t.computational_depth for t in input_tensors.values()), default=0)
            max_output_depth = max((t.computational_depth for t in output_tensors.values()), default=0)
            computational_depth = max(max_input_depth, max_output_depth)
            
            # Create the agentic kernel
            kernel = AgenticKernel(
                name=kernel_data['name'],
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                hypergraph=hypergraph,
                computational_depth=computational_depth,
                parallelism_pattern=kernel_data.get('parallelism_type', 'data_parallel'),
                metadata={
                    'source_file': analysis['filepath'],
                    'source_type': analysis['type'],
                    'line_number': kernel_data.get('line_number', 0)
                }
            )
            
            # Optimize for cognitive synergy
            kernel.optimize_for_cognitive_synergy()
            
            return kernel
            
        except Exception as e:
            print(f"Error creating agentic kernel from {kernel_data.get('name', 'unknown')}: {e}")
            return None
    
    def _estimate_computational_depth(self, dimensions: Tuple[int, ...]) -> int:
        """Estimate computational depth based on tensor dimensions."""
        if not dimensions:
            return 0
        
        # Simple heuristic: depth increases with dimensionality and size
        ndim = len(dimensions)
        size = max(dimensions) if dimensions else 1
        
        if ndim == 1:
            depth = 1
        elif ndim == 2:
            depth = 2
        else:
            depth = ndim
        
        # Increase depth for large tensors
        if size > 1000:
            depth += 1
        if size > 10000:
            depth += 1
        
        return depth
    
    def _create_hypergraph_pattern(self, kernel_data: Dict[str, Any], 
                                 input_tensors: Dict[str, TensorShape], 
                                 output_tensors: Dict[str, TensorShape]) -> HypergraphPattern:
        """Create hypergraph pattern for the kernel."""
        vertices = ['kernel'] + list(input_tensors.keys()) + list(output_tensors.keys())
        
        hypergraph = HypergraphPattern(vertices=vertices)
        
        # Add edges connecting kernel to inputs and outputs
        if input_tensors:
            input_edge = ('kernel',) + tuple(input_tensors.keys())
            hypergraph.add_edge(input_edge, weight=1.0)
        
        if output_tensors:
            output_edge = ('kernel',) + tuple(output_tensors.keys())
            hypergraph.add_edge(output_edge, weight=1.0)
        
        # Add edges between tensors based on dimensionality compatibility
        all_tensors = {**input_tensors, **output_tensors}
        tensor_names = list(all_tensors.keys())
        
        for i, name1 in enumerate(tensor_names):
            for name2 in tensor_names[i+1:]:
                shape1 = all_tensors[name1]
                shape2 = all_tensors[name2]
                
                # Connect tensors with compatible dimensions
                if self._tensors_compatible(shape1, shape2):
                    hypergraph.add_edge((name1, name2), weight=0.5)
        
        return hypergraph
    
    def _tensors_compatible(self, shape1: TensorShape, shape2: TensorShape) -> bool:
        """Check if two tensors have compatible dimensions for operations."""
        # Simple compatibility check
        dims1 = shape1.dimensions
        dims2 = shape2.dimensions
        
        # Same dimensionality
        if len(dims1) == len(dims2):
            return True
        
        # Matrix-vector compatibility
        if len(dims1) == 2 and len(dims2) == 1 and dims1[1] == dims2[0]:
            return True
        if len(dims2) == 2 and len(dims1) == 1 and dims2[1] == dims1[0]:
            return True
        
        return False


def extract_from_module(filepath: str) -> List[AgenticKernel]:
    """Convenience function to extract agentic kernels from a module."""
    factory = KernelFactory()
    return factory.extract_from_module(filepath)