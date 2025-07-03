"""Module for extracting dependencies from Python AST."""

import ast
from dataclasses import dataclass, field
from enum import Enum

from .graph_structure import GraphStructure
from .models import EdgeType


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module_name: str
    alias: str | None = None
    imported_names: list[str] = field(default_factory=list)
    is_relative: bool = False
    level: int = 0


@dataclass
class ReferenceInfo:
    """Information about a reference to a name."""
    name: str
    reference_type: 'ReferenceType'
    scope: str
    line: int
    column: int


class ReferenceType(Enum):
    """Types of references that can be tracked."""
    FUNCTION_CALL = "function_call"
    CLASS_INSTANTIATION = "class_instantiation"
    ATTRIBUTE_ACCESS = "attribute_access"
    VARIABLE_REFERENCE = "variable_reference"


class DependencyExtractor(ast.NodeVisitor):
    """Extract dependencies from Python AST."""

    def __init__(self):
        """Initialize the dependency extractor."""
        self.imports: dict[str, ImportInfo] = {}
        self.references: list[ReferenceInfo] = []
        self.scope_stack: list[str] = ["module"]
        self.graph: GraphStructure | None = None
        self.module_node_id: str | None = None
        self.node_map: dict[ast.AST, str] = {}

    def extract_dependencies(
        self,
        tree: ast.AST,
        graph: GraphStructure,
        module_node_id: str | None = None
    ) -> None:
        """Extract dependencies from AST and add them to the graph.

        Args:
            tree: The AST to extract dependencies from
            graph: The graph structure to add dependencies to
            module_node_id: Optional ID of the module node in the graph
        """
        self.graph = graph
        self.module_node_id = module_node_id

        # Build a map of AST nodes to graph node IDs for reference resolution
        self._build_node_map()

        # Visit the AST to extract dependencies
        self.visit(tree)

        # Add import edges to the graph
        if self.module_node_id:
            self._add_import_edges()

        # Add reference edges to the graph
        self._add_reference_edges()

    def _build_node_map(self) -> None:
        """Build a mapping from AST nodes to graph node IDs."""
        if not self.graph:
            return

        # Build a map of function/class names to their node IDs
        self.name_to_node_id: dict[str, str] = {}

        for node_id, node in self.graph.nodes.items():
            if node.node_type in ["FunctionDef", "AsyncFunctionDef", "ClassDef"]:
                # Extract name from AST node info
                name = node.ast_node_info.get("name")
                if name:
                    self.name_to_node_id[name] = node_id

        # Also map imported names to their import nodes
        for node_id, node in self.graph.nodes.items():
            if node.node_type == "alias":
                # This is an imported name
                name = node.ast_node_info.get("name")
                asname = node.ast_node_info.get("asname")
                # Use the alias if provided, otherwise use the original name
                import_name = asname if asname else name
                if import_name:
                    self.name_to_node_id[import_name] = node_id

    def _add_import_edges(self) -> None:
        """Add import edges to the graph."""
        if not self.graph or not self.module_node_id:
            return

        for import_name, import_info in self.imports.items():
            # Create a node for the imported module if it doesn't exist
            # Use import_name as part of the ID for determinism
            # Include the actual imported name to make ID unique
            import_node_id = f"import_{import_name}_{import_info.module_name or import_name}_{import_info.level}"

            from .models import ASTGraphEdge, ASTGraphNode

            # Check if node already exists
            if import_node_id not in self.graph.nodes:
                # Create node for imported module
                # For relative imports with no module name, use the import name
                label = import_info.module_name or import_name or f"relative_import_level_{import_info.level}"

                import_node = ASTGraphNode(
                    node_id=import_node_id,
                    node_type="ImportedModule",
                    label=label,
                    ast_node_info={
                        "module_name": import_info.module_name,
                        "alias": import_info.alias,
                        "is_relative": import_info.is_relative,
                        "level": import_info.level,
                        "imported_names": import_info.imported_names
                    }
                )

                # Add node to graph
                self.graph.add_node(import_node)

            # Create IMPORTS edge
            edge = ASTGraphEdge(
                edge_id=f"edge_{self.module_node_id}_to_{import_node_id}",
                source_id=self.module_node_id,
                target_id=import_node_id,
                edge_type=EdgeType.IMPORTS,
                label=f"imports {import_name}",
                metadata={"import_name": import_name}
            )

            # Add edge to graph
            self.graph.add_edge(edge)

    def _add_reference_edges(self) -> None:
        """Add reference edges to the graph."""
        if not self.graph:
            return

        from .models import ASTGraphEdge, ASTGraphNode

        # Group references by their containing function/class
        references_by_container: dict[str, list[ReferenceInfo]] = {}

        for ref in self.references:
            # Extract container name from scope
            if ref.scope != "module":
                # For "function:bar" or "class:MyClass", extract the name
                scope_parts = ref.scope.split(":")
                if len(scope_parts) == 2:
                    container_type, container_name = scope_parts
                    if container_name not in references_by_container:
                        references_by_container[container_name] = []
                    references_by_container[container_name].append(ref)
            else:
                # Module level references
                if "module" not in references_by_container:
                    references_by_container["module"] = []
                references_by_container["module"].append(ref)

        # Create edges for references
        for container_name, refs in references_by_container.items():
            # Find the source node (the function/class containing the reference)
            if container_name == "module":
                source_node_id = self.module_node_id
            else:
                source_node_id = self.name_to_node_id.get(container_name)

            if not source_node_id:
                continue

            for ref in refs:
                # Try to resolve the target node
                target_node_id = self.name_to_node_id.get(ref.name)

                # If target not found and it's an external reference (e.g., os.path.join)
                if not target_node_id and "." in ref.name:
                    # Create a node for external reference with deterministic ID
                    target_node_id = f"external_{ref.name}_{ref.line}_{ref.column}"
                    # Check if external node already exists
                    if target_node_id not in self.graph.nodes:
                        external_node = ASTGraphNode(
                            node_id=target_node_id,
                            node_type="ExternalReference",
                            label=ref.name,
                            ast_node_info={
                                "name": ref.name,
                                "reference_type": ref.reference_type.value
                            }
                        )
                        self.graph.add_node(external_node)

                if not target_node_id:
                    continue

                # Determine edge type
                if ref.reference_type == ReferenceType.FUNCTION_CALL:
                    edge_type = EdgeType.USES
                elif ref.reference_type == ReferenceType.CLASS_INSTANTIATION:
                    edge_type = EdgeType.INSTANTIATES
                else:
                    edge_type = EdgeType.USES

                # Create edge with deterministic ID
                edge = ASTGraphEdge(
                    edge_id=f"ref_{source_node_id}_to_{target_node_id}_{ref.line}_{ref.column}",
                    source_id=source_node_id,
                    target_id=target_node_id,
                    edge_type=edge_type,
                    label=f"{container_name} {edge_type.value} {ref.name}",
                    metadata={
                        "reference_type": ref.reference_type.value,
                        "line": ref.line,
                        "column": ref.column
                    }
                )

                try:
                    self.graph.add_edge(edge)
                except ValueError:
                    # Edge might already exist or nodes might not be in graph
                    pass

    def visit_Import(self, node: ast.Import) -> None:
        """Visit an import statement."""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imports[import_name] = ImportInfo(
                module_name=alias.name,
                alias=alias.asname,
                imported_names=[],
                is_relative=False,
                level=0
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit a from...import statement."""
        module_name = node.module or ""
        level = node.level or 0
        is_relative = level > 0

        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imports[import_name] = ImportInfo(
                module_name=module_name,
                alias=alias.asname,
                imported_names=[alias.name],
                is_relative=is_relative,
                level=level
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a function call."""
        # Determine if this is a function call or class instantiation
        if isinstance(node.func, ast.Name):
            name = node.func.id
            # Simple heuristic: if name starts with uppercase, it's likely a class
            ref_type = (ReferenceType.CLASS_INSTANTIATION
                       if name and name[0].isupper()
                       else ReferenceType.FUNCTION_CALL)

            self.references.append(ReferenceInfo(
                name=name,
                reference_type=ref_type,
                scope=self.scope_stack[-1],
                line=node.lineno,
                column=node.col_offset
            ))
        elif isinstance(node.func, ast.Attribute):
            # Handle attribute calls like os.path.join()
            full_name = self._get_full_attribute_name(node.func)
            self.references.append(ReferenceInfo(
                name=full_name,
                reference_type=ReferenceType.FUNCTION_CALL,
                scope=self.scope_stack[-1],
                line=node.lineno,
                column=node.col_offset
            ))

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit an attribute access."""
        # Only track attribute access if it's not part of a call
        if not isinstance(getattr(node, 'ctx', None), ast.Load):
            self.generic_visit(node)
            return

        full_name = self._get_full_attribute_name(node)
        self.references.append(ReferenceInfo(
            name=full_name,
            reference_type=ReferenceType.ATTRIBUTE_ACCESS,
            scope=self.scope_stack[-1],
            line=node.lineno,
            column=node.col_offset
        ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        self.scope_stack.append(f"function:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""
        self.scope_stack.append(f"function:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        self.scope_stack.append(f"class:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full name of an attribute access."""
        parts = [node.attr]
        current = node.value

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return ".".join(reversed(parts))
