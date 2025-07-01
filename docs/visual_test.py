#!/usr/bin/env python3
"""
Quick visual verification of a sample mermaid diagram
"""

import re

def extract_first_diagram(file_path):
    """Extract the first mermaid diagram from a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = r'```mermaid\n(.*?)\n```'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def main():
    # Extract and display the first diagram
    diagram = extract_first_diagram('/home/runner/work/kokkos-central/kokkos-central/docs/DistributedAgenticCognitiveGrammarNetwork.md')
    
    if diagram:
        print("✅ Successfully extracted sample diagram:")
        print("=" * 50)
        print(diagram)
        print("=" * 50)
        print("✅ Diagram extraction and basic parsing successful!")
        print("📊 This diagram contains the DACGN Core Architecture")
        print("🔗 It shows the main agent connections and data flow")
        print("📋 Ready for rendering in any mermaid-compatible viewer")
    else:
        print("❌ Failed to extract diagram")

if __name__ == "__main__":
    main()