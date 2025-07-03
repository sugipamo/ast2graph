#!/usr/bin/env python3
"""
Dependency analysis examples for ast2graph library.
"""

import os
from pathlib import Path
from collections import defaultdict
from ast2graph import parse_file, parse_directory


def example_extract_imports():
    """Example: Extract all imports from a file."""
    print("=== Extract Imports Example ===")
    
    code_file = Path(__file__).parent.parent / "src" / "ast2graph" / "api.py"
    
    if not code_file.exists():
        print(f"File not found: {code_file}")
        return
    
    # Parse with dependency extraction
    graph = parse_file(str(code_file), include_dependencies=True)
    
    print(f"Analyzing: {code_file.name}")
    
    # Find all import edges
    imports = []
    for edge in graph.edges.values():
        if edge.edge_type == "IMPORTS":
            source_node = graph.nodes[edge.source]
            target_node = graph.nodes[edge.target]
            imports.append((source_node, target_node))
    
    print(f"\nFound {len(imports)} imports:")
    for source, target in sorted(imports, key=lambda x: x[1].label):
        print(f"  - {target.label}")


def example_function_calls():
    """Example: Track function calls and usage."""
    print("\n=== Function Call Analysis Example ===")
    
    # Analyze the parser module
    parser_file = Path(__file__).parent.parent / "src" / "ast2graph" / "parser.py"
    
    if not parser_file.exists():
        print(f"File not found: {parser_file}")
        return
    
    graph = parse_file(str(parser_file), include_dependencies=True)
    
    # Find all function definitions
    functions = {
        node.id: node for node in graph.nodes.values()
        if node.ast_type == "FunctionDef"
    }
    
    print(f"Functions in {parser_file.name}:")
    for func in functions.values():
        print(f"  - {func.label}")
    
    # Find function usage
    print("\nFunction usage (USES edges):")
    uses_count = defaultdict(int)
    
    for edge in graph.edges.values():
        if edge.edge_type == "USES":
            target_node = graph.nodes[edge.target]
            if target_node.id in functions:
                uses_count[target_node.label] += 1
    
    for func_name, count in sorted(uses_count.items(), key=lambda x: -x[1]):
        print(f"  - {func_name}: called {count} times")


def example_class_instantiation():
    """Example: Find class instantiations."""
    print("\n=== Class Instantiation Example ===")
    
    # Analyze the graph builder module
    builder_file = Path(__file__).parent.parent / "src" / "ast2graph" / "graph_builder.py"
    
    if not builder_file.exists():
        print(f"File not found: {builder_file}")
        return
    
    graph = parse_file(str(builder_file), include_dependencies=True)
    
    # Find all class definitions
    classes = {
        node.label: node for node in graph.nodes.values()
        if node.ast_type == "ClassDef"
    }
    
    print(f"Classes in {builder_file.name}:")
    for class_name in sorted(classes.keys()):
        print(f"  - {class_name}")
    
    # Find instantiations
    print("\nClass instantiations:")
    instantiations = defaultdict(list)
    
    for edge in graph.edges.values():
        if edge.edge_type == "INSTANTIATES":
            source_node = graph.nodes[edge.source]
            target_node = graph.nodes[edge.target]
            if target_node.label in classes:
                instantiations[target_node.label].append(source_node)
    
    for class_name, instances in sorted(instantiations.items()):
        print(f"  {class_name}: {len(instances)} instantiation(s)")
        for inst in instances[:3]:  # Show first 3
            print(f"    - at {inst.label}")


def example_project_dependencies():
    """Example: Analyze dependencies across a project."""
    print("\n=== Project Dependencies Example ===")
    
    # Analyze the ast2graph source
    src_dir = Path(__file__).parent.parent / "src" / "ast2graph"
    
    if not src_dir.exists():
        print(f"Directory not found: {src_dir}")
        return
    
    # Parse all Python files
    graphs = parse_directory(str(src_dir), include_dependencies=True, file_pattern="*.py")
    
    print(f"Analyzing {len(graphs)} files in {src_dir.name}/")
    
    # Build import graph
    internal_imports = defaultdict(set)  # module -> set of imported modules
    external_imports = defaultdict(set)  # module -> set of external imports
    
    for file_path, graph in graphs.items():
        module_name = Path(file_path).stem
        
        for edge in graph.edges.values():
            if edge.edge_type == "IMPORTS":
                target_node = graph.nodes[edge.target]
                import_name = target_node.label
                
                # Check if it's an internal import
                if any(import_name.startswith(mod) for mod in graphs):
                    internal_imports[module_name].add(import_name)
                else:
                    external_imports[module_name].add(import_name)
    
    # Show internal dependencies
    print("\nInternal module dependencies:")
    for module, imports in sorted(internal_imports.items()):
        if imports:
            print(f"  {module}.py imports:")
            for imp in sorted(imports):
                print(f"    - {imp}")
    
    # Show external dependencies
    print("\nExternal dependencies used:")
    all_external = set()
    for imports in external_imports.values():
        all_external.update(imports)
    
    for imp in sorted(all_external):
        print(f"  - {imp}")


def example_dependency_graph_viz():
    """Example: Create a simple dependency visualization."""
    print("\n=== Dependency Visualization Example ===")
    
    src_dir = Path(__file__).parent.parent / "src" / "ast2graph"
    
    if not src_dir.exists():
        print(f"Directory not found: {src_dir}")
        return
    
    # Parse directory
    graphs = parse_directory(str(src_dir), include_dependencies=True, file_pattern="*.py")
    
    # Build simplified dependency matrix
    modules = sorted([Path(f).stem for f in graphs.keys()])
    dependencies = defaultdict(set)
    
    for file_path, graph in graphs.items():
        module = Path(file_path).stem
        
        # Find what this module imports
        for edge in graph.edges.values():
            if edge.edge_type == "IMPORTS":
                target_node = graph.nodes[edge.target]
                # Check if it's one of our modules
                for other_module in modules:
                    if target_node.label.startswith(other_module):
                        dependencies[module].add(other_module)
                        break
    
    # Print simple ASCII dependency matrix
    print(f"\nDependency matrix for {src_dir.name}:")
    print("(X = dependency, . = no dependency)")
    print()
    
    # Header
    print("              ", end="")
    for mod in modules[:8]:  # Limit to 8 for display
        print(f"{mod[:3]:>4}", end="")
    print()
    
    # Rows
    for from_mod in modules[:8]:
        print(f"{from_mod[:12]:>14}", end="")
        for to_mod in modules[:8]:
            if to_mod in dependencies.get(from_mod, set()):
                print("   X", end="")
            else:
                print("   .", end="")
        print()


def example_circular_dependencies():
    """Example: Detect potential circular dependencies."""
    print("\n=== Circular Dependency Detection Example ===")
    
    src_dir = Path(__file__).parent.parent / "src" / "ast2graph"
    
    if not src_dir.exists():
        print(f"Directory not found: {src_dir}")
        return
    
    # Parse directory
    graphs = parse_directory(str(src_dir), include_dependencies=True, file_pattern="*.py")
    
    # Build import graph
    import_graph = defaultdict(set)
    
    for file_path, graph in graphs.items():
        module = Path(file_path).stem
        
        for edge in graph.edges.values():
            if edge.edge_type == "IMPORTS":
                target_node = graph.nodes[edge.target]
                # Check if it's one of our modules
                for other_file in graphs:
                    other_module = Path(other_file).stem
                    if target_node.label.startswith(other_module):
                        import_graph[module].add(other_module)
    
    # Simple cycle detection (direct cycles only)
    print(f"Checking for circular dependencies in {len(graphs)} modules...")
    
    cycles_found = []
    for module_a, imports_a in import_graph.items():
        for module_b in imports_a:
            if module_a in import_graph.get(module_b, set()):
                cycle = tuple(sorted([module_a, module_b]))
                if cycle not in cycles_found:
                    cycles_found.append(cycle)
    
    if cycles_found:
        print(f"\nFound {len(cycles_found)} circular dependencies:")
        for module_a, module_b in cycles_found:
            print(f"  - {module_a} <-> {module_b}")
    else:
        print("\nNo circular dependencies found! ✓")


if __name__ == "__main__":
    # Run all examples
    example_extract_imports()
    example_function_calls()
    example_class_instantiation()
    example_project_dependencies()
    example_dependency_graph_viz()
    example_circular_dependencies()