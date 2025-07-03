#!/usr/bin/env python3
"""
Basic usage examples for ast2graph library.
"""

from ast2graph import parse_file, parse_code, parse_directory


def example_parse_file():
    """Example: Parse a single Python file."""
    print("=== Parse Single File Example ===")
    
    # Parse this script itself
    graph = parse_file(__file__)
    
    print(f"File: {__file__}")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")
    
    # Show first 5 nodes
    print("\nFirst 5 nodes:")
    for i, (node_id, node) in enumerate(graph.nodes.items()):
        if i >= 5:
            break
        print(f"  - {node.label} (type: {node.ast_type})")
    
    # Show edge type distribution
    edge_types = {}
    for edge in graph.edges.values():
        edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
    
    print("\nEdge type distribution:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  - {edge_type}: {count}")


def example_parse_code():
    """Example: Parse Python code from string."""
    print("\n=== Parse Code String Example ===")
    
    code = """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

calc = Calculator()
result = calc.add(5, 3)
print(result)
"""
    
    graph = parse_code(code, filename="calculator.py")
    
    print(f"Code length: {len(code)} characters")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")
    
    # Find all function definitions
    functions = [
        node for node in graph.nodes.values()
        if node.ast_type == "FunctionDef"
    ]
    
    print(f"\nFunctions found: {len(functions)}")
    for func in functions:
        print(f"  - {func.label}")


def example_parse_with_dependencies():
    """Example: Parse with dependency extraction."""
    print("\n=== Parse with Dependencies Example ===")
    
    code = """
import os
from pathlib import Path
import json

def read_config(config_path):
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}
"""
    
    # Parse without dependencies
    graph_no_deps = parse_code(code, include_dependencies=False)
    
    # Parse with dependencies
    graph_with_deps = parse_code(code, include_dependencies=True)
    
    print("Without dependencies:")
    print(f"  Nodes: {len(graph_no_deps.nodes)}")
    print(f"  Edges: {len(graph_no_deps.edges)}")
    
    print("\nWith dependencies:")
    print(f"  Nodes: {len(graph_with_deps.nodes)}")
    print(f"  Edges: {len(graph_with_deps.edges)}")
    
    # Find import edges
    import_edges = [
        edge for edge in graph_with_deps.edges.values()
        if edge.edge_type == "IMPORTS"
    ]
    
    print(f"\nImport relationships found: {len(import_edges)}")
    for edge in import_edges:
        source = graph_with_deps.nodes[edge.source]
        target = graph_with_deps.nodes[edge.target]
        print(f"  - {source.label} imports {target.label}")


def example_graph_navigation():
    """Example: Navigate the graph structure."""
    print("\n=== Graph Navigation Example ===")
    
    code = """
def parent_function():
    def nested_function():
        x = 10
        return x * 2
    
    result = nested_function()
    return result
"""
    
    graph = parse_code(code)
    
    # Find the parent function
    parent_func = next(
        node for node in graph.nodes.values()
        if node.label == "parent_function"
    )
    
    print(f"Found function: {parent_func.label}")
    
    # Get all children of the parent function
    children = graph.get_children(parent_func.id)
    print(f"\nChildren of {parent_func.label}:")
    for child in children:
        print(f"  - {child.label} (type: {child.ast_type})")
    
    # Find nested function and get its parent
    nested_func = next(
        node for node in graph.nodes.values()
        if node.label == "nested_function"
    )
    
    parents = graph.get_parents(nested_func.id)
    print(f"\nParents of {nested_func.label}:")
    for parent in parents:
        print(f"  - {parent.label} (type: {parent.ast_type})")


def example_export_graph():
    """Example: Export graph to different formats."""
    print("\n=== Export Graph Example ===")
    
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    graph = parse_code(code, include_dependencies=True)
    
    # Export to dictionary
    graph_dict = graph.to_dict()
    print(f"Exported to dict with {len(graph_dict['nodes'])} nodes")
    
    # Export to JSON string
    json_str = graph.to_json(indent=2)
    print(f"\nJSON export preview (first 200 chars):")
    print(json_str[:200] + "...")
    
    # Validate graph integrity
    validation = graph.validate()
    print(f"\nGraph validation: {'✓ Valid' if validation.is_valid else '✗ Invalid'}")
    if not validation.is_valid:
        for error in validation.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    # Run all examples
    example_parse_file()
    example_parse_code()
    example_parse_with_dependencies()
    example_graph_navigation()
    example_export_graph()