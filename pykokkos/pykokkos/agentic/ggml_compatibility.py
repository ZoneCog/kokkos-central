"""
GGML compatibility layer for agentic kernels.

This module provides functionality to convert agentic kernels to GGML-compatible
representations and optimize for future integration with GGML-based systems.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Note: For standalone testing, we'll import relatively later
# from .kernel_dict import AgenticKernel, TensorShape, TensorLayout


@dataclass
class GGMLTensorSpec:
    """GGML-compatible tensor specification."""
    name: str
    shape: List[int]
    dtype: str
    layout: str
    quantization: Optional[str] = None
    memory_type: str = "host"


@dataclass
class GGMLOperatorSpec:
    """GGML-compatible operator specification."""
    name: str
    type: str
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, Any]


@dataclass
class GGMLGraphSpec:
    """GGML-compatible computational graph specification."""
    name: str
    tensors: List[GGMLTensorSpec]
    operators: List[GGMLOperatorSpec]
    metadata: Dict[str, Any]


class GGMLAdapter:
    """
    Adapter for converting agentic kernels to GGML-compatible formats.
    
    Provides functionality to translate agentic kernel representations into
    formats that can be easily integrated with GGML-based systems.
    """
    
    def __init__(self):
        self.dtype_mapping = {
            'f32': 'GGML_TYPE_F32',
            'f16': 'GGML_TYPE_F16',
            'q4_0': 'GGML_TYPE_Q4_0',
            'q4_1': 'GGML_TYPE_Q4_1',
            'q8_0': 'GGML_TYPE_Q8_0',
            'i32': 'GGML_TYPE_I32',
            'i16': 'GGML_TYPE_I16',
            'i8': 'GGML_TYPE_I8'
        }
        
        self.layout_mapping = {
            TensorLayout.ROW_MAJOR: 'GGML_LAYOUT_ROW_MAJOR',
            TensorLayout.COL_MAJOR: 'GGML_LAYOUT_COL_MAJOR',
            TensorLayout.SPARSE_COO: 'GGML_LAYOUT_SPARSE_COO',
            TensorLayout.SPARSE_CSR: 'GGML_LAYOUT_SPARSE_CSR',
            TensorLayout.HYPERGRAPH: 'GGML_LAYOUT_CUSTOM'
        }
        
        self.operation_mapping = {
            'parallel_for': 'ggml_map',
            'parallel_reduce': 'ggml_reduce',
            'parallel_scan': 'ggml_scan',
            'matrix_multiply': 'ggml_mul_mat',
            'elementwise_add': 'ggml_add',
            'elementwise_mul': 'ggml_mul',
            'convolution': 'ggml_conv_2d',
            'transpose': 'ggml_transpose'
        }
    
    def convert_kernel_to_ggml(self, kernel: AgenticKernel) -> GGMLGraphSpec:
        """
        Convert an agentic kernel to GGML-compatible graph specification.
        
        Args:
            kernel: The agentic kernel to convert
            
        Returns:
            GGML-compatible graph specification
        """
        # Convert tensors
        ggml_tensors = []
        
        # Convert input tensors
        for name, tensor in kernel.input_tensors.items():
            ggml_tensor = self._convert_tensor_to_ggml(name, tensor, is_input=True)
            ggml_tensors.append(ggml_tensor)
        
        # Convert output tensors
        for name, tensor in kernel.output_tensors.items():
            ggml_tensor = self._convert_tensor_to_ggml(name, tensor, is_input=False)
            ggml_tensors.append(ggml_tensor)
        
        # Convert kernel operation to GGML operators
        ggml_operators = self._convert_kernel_operations_to_ggml(kernel)
        
        # Create GGML graph specification
        ggml_graph = GGMLGraphSpec(
            name=kernel.name,
            tensors=ggml_tensors,
            operators=ggml_operators,
            metadata={
                'original_kernel': kernel.name,
                'computational_depth': kernel.computational_depth,
                'parallelism_pattern': kernel.parallelism_pattern,
                'memory_requirements': kernel.memory_requirements,
                'total_dof': kernel.calculate_total_dof(),
                'cognitive_affinity': kernel.hypergraph.cognitive_affinity,
                'ggml_version': '0.1.0'
            }
        )
        
        return ggml_graph
    
    def _convert_tensor_to_ggml(self, name: str, tensor: TensorShape, is_input: bool) -> GGMLTensorSpec:
        """Convert a tensor shape to GGML tensor specification."""
        # Handle dynamic dimensions
        shape = []
        for dim in tensor.dimensions:
            if dim <= 0:  # Dynamic dimension
                shape.append(-1)  # GGML uses -1 for dynamic dimensions
            else:
                shape.append(dim)
        
        # Determine data type (default to f32 if not specified)
        dtype = self.dtype_mapping.get('f32', 'GGML_TYPE_F32')
        
        # Convert layout
        layout = self.layout_mapping.get(tensor.layout, 'GGML_LAYOUT_ROW_MAJOR')
        
        # Determine quantization based on computational depth and size
        quantization = None
        total_elements = tensor.degrees_of_freedom
        if total_elements > 100000 and tensor.computational_depth <= 2:
            # Use quantization for large, simple tensors
            quantization = 'GGML_TYPE_Q4_0'
        
        # Determine memory type
        memory_type = "device" if total_elements > 10000 else "host"
        
        return GGMLTensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            layout=layout,
            quantization=quantization,
            memory_type=memory_type
        )
    
    def _convert_kernel_operations_to_ggml(self, kernel: AgenticKernel) -> List[GGMLOperatorSpec]:
        """Convert kernel operations to GGML operators."""
        operators = []
        
        # Determine operation type based on kernel characteristics
        operation_type = self._infer_operation_type(kernel)
        
        # Create main operation
        main_op = GGMLOperatorSpec(
            name=f"{kernel.name}_main",
            type=self.operation_mapping.get(operation_type, 'ggml_map'),
            inputs=list(kernel.input_tensors.keys()),
            outputs=list(kernel.output_tensors.keys()),
            parameters=self._generate_operation_parameters(kernel, operation_type)
        )
        operators.append(main_op)
        
        # Add auxiliary operations based on hypergraph patterns
        aux_ops = self._generate_auxiliary_operations(kernel)
        operators.extend(aux_ops)
        
        return operators
    
    def _infer_operation_type(self, kernel: AgenticKernel) -> str:
        """Infer the primary operation type from kernel characteristics."""
        # Check parallelism pattern
        if kernel.parallelism_pattern == 'reduction':
            return 'parallel_reduce'
        elif kernel.parallelism_pattern == 'data_parallel':
            return 'parallel_for'
        
        # Check tensor shapes for matrix operations
        input_shapes = [t.dimensions for t in kernel.input_tensors.values()]
        output_shapes = [t.dimensions for t in kernel.output_tensors.values()]
        
        # Matrix multiplication pattern
        if (len(input_shapes) >= 2 and len(output_shapes) >= 1 and
            all(len(shape) == 2 for shape in input_shapes[:2]) and
            len(output_shapes[0]) == 2):
            return 'matrix_multiply'
        
        # Element-wise operations
        if (len(input_shapes) >= 2 and len(output_shapes) >= 1 and
            input_shapes[0] == input_shapes[1] == output_shapes[0]):
            return 'elementwise_add'
        
        # Default to parallel_for
        return 'parallel_for'
    
    def _generate_operation_parameters(self, kernel: AgenticKernel, operation_type: str) -> Dict[str, Any]:
        """Generate operation parameters for GGML operators."""
        parameters = {
            'computational_depth': kernel.computational_depth,
            'memory_requirements': kernel.memory_requirements
        }
        
        if operation_type == 'matrix_multiply':
            parameters.update({
                'transpose_a': False,
                'transpose_b': False,
                'alpha': 1.0,
                'beta': 0.0
            })
        elif operation_type in ['parallel_reduce', 'ggml_reduce']:
            parameters.update({
                'reduction_type': 'sum',
                'axis': -1,
                'keepdims': False
            })
        elif operation_type == 'convolution':
            parameters.update({
                'stride': [1, 1],
                'padding': [0, 0],
                'dilation': [1, 1]
            })
        
        return parameters
    
    def _generate_auxiliary_operations(self, kernel: AgenticKernel) -> List[GGMLOperatorSpec]:
        """Generate auxiliary operations based on hypergraph patterns."""
        aux_ops = []
        
        # Add operations for hypergraph connections
        for hyperedge in kernel.hypergraph.hyperedges:
            if len(hyperedge) > 2:  # Multi-way connection
                # Create auxiliary operation for multi-way interaction
                aux_op = GGMLOperatorSpec(
                    name=f"{kernel.name}_aux_{len(aux_ops)}",
                    type='ggml_custom',
                    inputs=list(hyperedge),
                    outputs=[f"{kernel.name}_aux_out_{len(aux_ops)}"],
                    parameters={
                        'operation': 'hypergraph_interaction',
                        'weight': kernel.hypergraph.edge_weights.get(hyperedge, 1.0),
                        'cognitive_affinity': kernel.hypergraph.cognitive_affinity
                    }
                )
                aux_ops.append(aux_op)
        
        return aux_ops
    
    def export_to_ggml_json(self, ggml_graph: GGMLGraphSpec) -> str:
        """Export GGML graph specification to JSON format."""
        graph_dict = {
            'name': ggml_graph.name,
            'tensors': [
                {
                    'name': tensor.name,
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'layout': tensor.layout,
                    'quantization': tensor.quantization,
                    'memory_type': tensor.memory_type
                }
                for tensor in ggml_graph.tensors
            ],
            'operators': [
                {
                    'name': op.name,
                    'type': op.type,
                    'inputs': op.inputs,
                    'outputs': op.outputs,
                    'parameters': op.parameters
                }
                for op in ggml_graph.operators
            ],
            'metadata': ggml_graph.metadata
        }
        
        return json.dumps(graph_dict, indent=2)
    
    def generate_ggml_c_code(self, ggml_graph: GGMLGraphSpec) -> str:
        """Generate C code for GGML implementation."""
        code_lines = [
            '#include "ggml.h"',
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '',
            f'// Generated GGML code for kernel: {ggml_graph.name}',
            '',
            f'struct ggml_context* create_{ggml_graph.name}_context() {{',
            '    size_t ctx_size = ggml_tensor_overhead() * 1024;',
            '    struct ggml_init_params params = {',
            '        .mem_size = ctx_size,',
            '        .mem_buffer = NULL,',
            '        .no_alloc = false',
            '    };',
            '    return ggml_init(params);',
            '}',
            ''
        ]
        
        # Generate tensor creation code
        code_lines.append(f'void create_{ggml_graph.name}_tensors(struct ggml_context* ctx) {{')
        
        for tensor in ggml_graph.tensors:
            shape_str = ', '.join(map(str, tensor.shape))
            code_lines.append(f'    // Create tensor: {tensor.name}')
            code_lines.append(f'    struct ggml_tensor* {tensor.name} = ggml_new_tensor(ctx, {tensor.dtype}, {len(tensor.shape)}, (int64_t[]){{ {shape_str} }});')
            code_lines.append('')
        
        code_lines.append('}')
        code_lines.append('')
        
        # Generate computation code
        code_lines.append(f'struct ggml_tensor* compute_{ggml_graph.name}(struct ggml_context* ctx) {{')
        
        for op in ggml_graph.operators:
            code_lines.append(f'    // Operation: {op.name}')
            if op.type == 'ggml_mul_mat':
                code_lines.append(f'    struct ggml_tensor* {op.outputs[0]} = ggml_mul_mat(ctx, {op.inputs[0]}, {op.inputs[1]});')
            elif op.type == 'ggml_add':
                code_lines.append(f'    struct ggml_tensor* {op.outputs[0]} = ggml_add(ctx, {op.inputs[0]}, {op.inputs[1]});')
            elif op.type == 'ggml_map':
                code_lines.append(f'    struct ggml_tensor* {op.outputs[0]} = ggml_map_custom1(ctx, {op.inputs[0]}, custom_kernel, NULL);')
            else:
                code_lines.append(f'    // Custom operation: {op.type}')
                code_lines.append(f'    struct ggml_tensor* {op.outputs[0]} = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);')
            code_lines.append('')
        
        if ggml_graph.operators:
            last_output = ggml_graph.operators[-1].outputs[0] if ggml_graph.operators[-1].outputs else 'NULL'
            code_lines.append(f'    return {last_output};')
        else:
            code_lines.append('    return NULL;')
        
        code_lines.append('}')
        
        return '\n'.join(code_lines)
    
    def optimize_for_ggml(self, kernel: AgenticKernel) -> AgenticKernel:
        """Optimize agentic kernel specifically for GGML compatibility."""
        # Create optimized copy
        optimized_kernel = AgenticKernel(
            name=kernel.name,
            input_tensors=kernel.input_tensors.copy(),
            output_tensors=kernel.output_tensors.copy(),
            hypergraph=kernel.hypergraph,
            computational_depth=kernel.computational_depth,
            parallelism_pattern=kernel.parallelism_pattern,
            memory_requirements=kernel.memory_requirements,
            ggml_compatible=True,
            metadata=kernel.metadata.copy()
        )
        
        # GGML-specific optimizations
        
        # 1. Ensure tensor shapes are GGML-compatible
        for tensor in list(optimized_kernel.input_tensors.values()) + list(optimized_kernel.output_tensors.values()):
            # GGML prefers certain dimension alignments
            if len(tensor.dimensions) >= 2:
                # Ensure dimensions are multiples of 32 for better SIMD performance
                new_dims = []
                for dim in tensor.dimensions:
                    if dim > 0 and dim % 32 != 0:
                        new_dims.append(((dim + 31) // 32) * 32)
                    else:
                        new_dims.append(dim)
                tensor.dimensions = tuple(new_dims)
                tensor.__post_init__()  # Recalculate DOF
        
        # 2. Optimize for GGML's preferred layouts
        for tensor in list(optimized_kernel.input_tensors.values()) + list(optimized_kernel.output_tensors.values()):
            # GGML typically prefers row-major layout
            if tensor.layout != TensorLayout.ROW_MAJOR:
                tensor.layout = TensorLayout.ROW_MAJOR
        
        # 3. Adjust computational depth for GGML's execution model
        if optimized_kernel.computational_depth > 4:
            optimized_kernel.computational_depth = 4  # GGML works best with moderate depth
        
        # 4. Update metadata
        optimized_kernel.metadata['ggml_optimized'] = True
        optimized_kernel.metadata['ggml_version_target'] = '0.1.0'
        
        return optimized_kernel
    
    def validate_ggml_compatibility(self, kernel: AgenticKernel) -> Dict[str, Any]:
        """Validate kernel compatibility with GGML and provide recommendations."""
        issues = []
        recommendations = []
        compatibility_score = 1.0
        
        # Check tensor shapes
        for name, tensor in {**kernel.input_tensors, **kernel.output_tensors}.items():
            if len(tensor.dimensions) > 4:
                issues.append(f"Tensor {name} has {len(tensor.dimensions)} dimensions, GGML typically supports up to 4D")
                compatibility_score -= 0.1
                recommendations.append(f"Consider reshaping tensor {name} to 4D or lower")
            
            if any(d > 100000 for d in tensor.dimensions if d > 0):
                issues.append(f"Tensor {name} has very large dimensions, may cause memory issues in GGML")
                compatibility_score -= 0.05
                recommendations.append(f"Consider tiling or chunking tensor {name}")
        
        # Check computational depth
        if kernel.computational_depth > 6:
            issues.append(f"Computational depth {kernel.computational_depth} is high for GGML")
            compatibility_score -= 0.1
            recommendations.append("Consider reducing computational depth through optimization")
        
        # Check hypergraph complexity
        if len(kernel.hypergraph.hyperedges) > 20:
            issues.append("Very complex hypergraph may not translate well to GGML's linear graph model")
            compatibility_score -= 0.05
            recommendations.append("Consider simplifying hypergraph structure")
        
        compatibility_score = max(0.0, compatibility_score)
        
        return {
            'compatible': len(issues) == 0,
            'compatibility_score': compatibility_score,
            'issues': issues,
            'recommendations': recommendations,
            'ggml_ready': compatibility_score > 0.8
        }