#!/usr/bin/env python3
"""
DACGN Diagram Validation and Testing Framework

This script validates all mermaid diagrams in the DACGN documentation
for syntax correctness, semantic accuracy, and visual clarity.
"""

import re
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class DiagramType(Enum):
    GRAPH = "graph"
    FLOWCHART = "flowchart"
    SEQUENCE = "sequenceDiagram"
    CLASS = "classDiagram"
    STATE = "stateDiagram"
    GITGRAPH = "gitGraph"

@dataclass
class DiagramInfo:
    content: str
    type: DiagramType
    start_line: int
    end_line: int
    file_path: str

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class MermaidValidator:
    """Validates mermaid diagram syntax and semantics"""
    
    def __init__(self):
        self.syntax_patterns = {
            DiagramType.GRAPH: [
                r'graph\s+(TB|TD|BT|RL|LR)',
                r'subgraph\s+[\"\w\s]+',
                r'\w+\s*-->\s*\w+',
                r'\w+\s*\-\.\->\s*\w+',
                r'\[\w+.*\]',
                r'\(\w+.*\)',
                r'\{\w+.*\}'
            ],
            DiagramType.FLOWCHART: [
                r'flowchart\s+(TB|TD|BT|RL|LR)',
                r'subgraph\s+[\"\w\s]+',
                r'\w+\s*-->\s*\w+',
                r'\[\w+.*\]',
                r'\(\w+.*\)',
                r'\{\w+.*\}'
            ],
            DiagramType.SEQUENCE: [
                r'sequenceDiagram',
                r'participant\s+\w+\s+as\s+.+',
                r'\w+->>.*:\s*.+',
                r'Note\s+(left|right)\s+of\s+\w+:\s*.+'
            ],
            DiagramType.CLASS: [
                r'classDiagram',
                r'class\s+\w+\s*\{',
                r'\+\w+\s*\w*',
                r'\-\w+\s*\w*',
                r'\w+\s*<\|\-\-\s*\w+',
                r'\w+\s*-->\s*\w+'
            ],
            DiagramType.STATE: [
                r'stateDiagram(-v2)?',
                r'\[\*\]\s*-->\s*\w+',
                r'\w+\s*-->\s*\w+\s*:\s*.+',
                r'\w+\s*:\s*.+'
            ]
        }
    
    def extract_diagrams(self, file_path: str) -> List[DiagramInfo]:
        """Extract all mermaid diagrams from a markdown file"""
        diagrams = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return diagrams
        
        # Find all mermaid code blocks
        pattern = r'```mermaid\n(.*?)\n```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            diagram_content = match.group(1).strip()
            start_pos = match.start()
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + 1
            end_line = start_line + diagram_content.count('\n')
            
            # Determine diagram type
            diagram_type = self._identify_diagram_type(diagram_content)
            
            diagrams.append(DiagramInfo(
                content=diagram_content,
                type=diagram_type,
                start_line=start_line,
                end_line=end_line,
                file_path=file_path
            ))
        
        return diagrams
    
    def _identify_diagram_type(self, content: str) -> DiagramType:
        """Identify the type of mermaid diagram"""
        content_lower = content.lower().strip()
        
        if content_lower.startswith('graph'):
            return DiagramType.GRAPH
        elif content_lower.startswith('flowchart'):
            return DiagramType.FLOWCHART
        elif content_lower.startswith('sequencediagram'):
            return DiagramType.SEQUENCE
        elif content_lower.startswith('classdiagram'):
            return DiagramType.CLASS
        elif content_lower.startswith('statediagram'):
            return DiagramType.STATE
        else:
            return DiagramType.GRAPH  # Default assumption
    
    def validate_diagram(self, diagram: DiagramInfo) -> ValidationResult:
        """Validate a single mermaid diagram"""
        errors = []
        warnings = []
        suggestions = []
        
        # Basic syntax validation
        syntax_errors = self._validate_syntax(diagram)
        errors.extend(syntax_errors)
        
        # Semantic validation
        semantic_warnings = self._validate_semantics(diagram)
        warnings.extend(semantic_warnings)
        
        # Visual clarity suggestions
        clarity_suggestions = self._analyze_clarity(diagram)
        suggestions.extend(clarity_suggestions)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_syntax(self, diagram: DiagramInfo) -> List[str]:
        """Validate diagram syntax"""
        errors = []
        lines = diagram.content.split('\n')
        
        # Check for basic syntax elements
        patterns = self.syntax_patterns.get(diagram.type, [])
        
        # Validate braces/brackets matching
        open_brackets = {'[': 0, '(': 0, '{': 0}
        close_brackets = {']': 0, ')': 0, '}': 0}
        bracket_map = {'[': ']', '(': ')', '{': '}'}
        
        for line_num, line in enumerate(lines, 1):
            for char in line:
                if char in open_brackets:
                    open_brackets[char] += 1
                elif char in close_brackets:
                    close_brackets[char] += 1
        
        # Check bracket matching
        for open_bracket, close_bracket in bracket_map.items():
            if open_brackets[open_bracket] != close_brackets[close_bracket]:
                errors.append(f"Mismatched {open_bracket}{close_bracket} brackets")
        
        # Check for common syntax errors
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('%'):
                continue
            
            # Skip validation for state diagram transition syntax and special mermaid syntax
            if diagram.type == DiagramType.STATE and (':' in line or '-->' in line):
                continue
            
            # Skip class diagram relationship syntax
            if diagram.type == DiagramType.CLASS and ('-->' in line or '<|--' in line):
                continue
                
            # Allow dotted arrows and special mermaid syntax
            if '-.->' in line or '<-->' in line or '==>' in line:
                continue
                
            # Check for unterminated strings
            if line.count('"') % 2 != 0:
                errors.append(f"Line {line_num}: Unterminated string")
        
        return errors
    
    def _validate_semantics(self, diagram: DiagramInfo) -> List[str]:
        """Validate diagram semantics"""
        warnings = []
        
        # Extract nodes and connections
        nodes = set()
        connections = []
        
        lines = diagram.content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract node definitions
            node_matches = re.findall(r'(\w+)\s*\[', line)
            nodes.update(node_matches)
            
            # Extract connections
            connection_matches = re.findall(r'(\w+)\s*-->\s*(\w+)', line)
            connections.extend(connection_matches)
            
            dotted_matches = re.findall(r'(\w+)\s*\-\.\->\s*(\w+)', line)
            connections.extend(dotted_matches)
        
        # Check for orphaned nodes
        connected_nodes = set()
        for source, target in connections:
            connected_nodes.add(source)
            connected_nodes.add(target)
        
        orphaned = nodes - connected_nodes
        if orphaned:
            warnings.append(f"Orphaned nodes detected: {', '.join(orphaned)}")
        
        # Check for circular dependencies in simple cases
        if self._has_simple_cycles(connections):
            warnings.append("Potential circular dependencies detected")
        
        return warnings
    
    def _has_simple_cycles(self, connections: List[Tuple[str, str]]) -> bool:
        """Detect simple cycles in connections"""
        graph = {}
        for source, target in connections:
            if source not in graph:
                graph[source] = []
            graph[source].append(target)
        
        # Simple DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _analyze_clarity(self, diagram: DiagramInfo) -> List[str]:
        """Analyze diagram for visual clarity"""
        suggestions = []
        
        lines = diagram.content.split('\n')
        
        # Count nodes and connections
        node_count = len(re.findall(r'\w+\s*\[', diagram.content))
        connection_count = len(re.findall(r'-->', diagram.content))
        
        # Suggest organization improvements
        if node_count > 20:
            suggestions.append("Consider breaking large diagram into smaller subgraphs")
        
        if connection_count > 30:
            suggestions.append("High number of connections may reduce clarity")
        
        # Check for subgraph usage
        if node_count > 10 and 'subgraph' not in diagram.content:
            suggestions.append("Consider using subgraphs to organize related nodes")
        
        # Check for descriptive labels
        short_labels = re.findall(r'\[(\w{1,2})\]', diagram.content)
        if len(short_labels) > 3:
            suggestions.append("Consider using more descriptive node labels")
        
        return suggestions

class TestRunner:
    """Runs comprehensive tests on DACGN documentation"""
    
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.validator = MermaidValidator()
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("🔍 Starting DACGN Documentation Validation")
        print("=" * 50)
        
        # Find all markdown files
        md_files = list(self.docs_dir.glob("*.md"))
        
        total_diagrams = 0
        valid_diagrams = 0
        
        for md_file in md_files:
            print(f"\n📄 Validating file: {md_file.name}")
            
            diagrams = self.validator.extract_diagrams(str(md_file))
            file_results = []
            
            for i, diagram in enumerate(diagrams, 1):
                print(f"  📊 Diagram {i} (Type: {diagram.type.value})")
                
                result = self.validator.validate_diagram(diagram)
                total_diagrams += 1
                
                if result.is_valid:
                    valid_diagrams += 1
                    print(f"    ✅ Valid")
                else:
                    print(f"    ❌ Invalid")
                
                if result.errors:
                    print(f"    🚨 Errors: {len(result.errors)}")
                    for error in result.errors:
                        print(f"      - {error}")
                
                if result.warnings:
                    print(f"    ⚠️  Warnings: {len(result.warnings)}")
                    for warning in result.warnings:
                        print(f"      - {warning}")
                
                if result.suggestions:
                    print(f"    💡 Suggestions: {len(result.suggestions)}")
                    for suggestion in result.suggestions:
                        print(f"      - {suggestion}")
                
                file_results.append({
                    'diagram_type': diagram.type.value,
                    'start_line': diagram.start_line,
                    'end_line': diagram.end_line,
                    'is_valid': result.is_valid,
                    'errors': result.errors,
                    'warnings': result.warnings,
                    'suggestions': result.suggestions
                })
            
            self.results[str(md_file)] = file_results
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(md_files)}")
        print(f"Total diagrams found: {total_diagrams}")
        print(f"Valid diagrams: {valid_diagrams}")
        print(f"Invalid diagrams: {total_diagrams - valid_diagrams}")
        
        if total_diagrams > 0:
            success_rate = (valid_diagrams / total_diagrams) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
            if success_rate >= 95:
                print("🎉 Excellent! Documentation quality is very high.")
            elif success_rate >= 80:
                print("👍 Good! Minor improvements needed.")
            elif success_rate >= 60:
                print("⚠️  Fair. Several issues need attention.")
            else:
                print("🚨 Poor. Significant improvements required.")
        
        return {
            'total_files': len(md_files),
            'total_diagrams': total_diagrams,
            'valid_diagrams': valid_diagrams,
            'success_rate': (valid_diagrams / total_diagrams * 100) if total_diagrams > 0 else 0,
            'detailed_results': self.results
        }
    
    def generate_report(self, output_file: str = None):
        """Generate a detailed HTML report"""
        if not output_file:
            output_file = self.docs_dir / "validation_report.html"
        
        html_content = self._create_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Detailed report saved to: {output_file}")
    
    def _create_html_report(self) -> str:
        """Create HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DACGN Documentation Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .file-section { margin: 20px 0; border: 1px solid #ddd; padding: 15px; }
                .diagram { margin: 10px 0; padding: 10px; background: #f9f9f9; }
                .valid { border-left: 5px solid #4CAF50; }
                .invalid { border-left: 5px solid #f44336; }
                .error { color: #d32f2f; }
                .warning { color: #f57c00; }
                .suggestion { color: #1976d2; }
                ul { margin: 5px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DACGN Documentation Validation Report</h1>
                <p>Generated on: """ + str(Path().absolute()) + """</p>
            </div>
        """
        
        for file_path, diagrams in self.results.items():
            html += f"""
            <div class="file-section">
                <h2>📄 {Path(file_path).name}</h2>
                <p>Total diagrams: {len(diagrams)}</p>
            """
            
            for i, diagram in enumerate(diagrams, 1):
                status_class = "valid" if diagram['is_valid'] else "invalid"
                status_icon = "✅" if diagram['is_valid'] else "❌"
                
                html += f"""
                <div class="diagram {status_class}">
                    <h3>{status_icon} Diagram {i} - {diagram['diagram_type']}</h3>
                    <p>Lines: {diagram['start_line']}-{diagram['end_line']}</p>
                """
                
                if diagram['errors']:
                    html += "<h4 class='error'>🚨 Errors:</h4><ul>"
                    for error in diagram['errors']:
                        html += f"<li class='error'>{error}</li>"
                    html += "</ul>"
                
                if diagram['warnings']:
                    html += "<h4 class='warning'>⚠️ Warnings:</h4><ul>"
                    for warning in diagram['warnings']:
                        html += f"<li class='warning'>{warning}</li>"
                    html += "</ul>"
                
                if diagram['suggestions']:
                    html += "<h4 class='suggestion'>💡 Suggestions:</h4><ul>"
                    for suggestion in diagram['suggestions']:
                        html += f"<li class='suggestion'>{suggestion}</li>"
                    html += "</ul>"
                
                html += "</div>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html

def main():
    """Main entry point"""
    docs_dir = "/home/runner/work/kokkos-central/kokkos-central/docs"
    
    if not os.path.exists(docs_dir):
        print(f"❌ Error: Documentation directory not found: {docs_dir}")
        sys.exit(1)
    
    # Run tests
    runner = TestRunner(docs_dir)
    results = runner.run_all_tests()
    
    # Generate report
    runner.generate_report()
    
    # Exit with appropriate code
    if results['success_rate'] >= 80:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()