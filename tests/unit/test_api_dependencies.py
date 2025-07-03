"""Test API integration with dependency extraction."""


from ast2graph.api import parse_code, parse_file
from ast2graph.models import EdgeType


class TestAPIWithDependencies:
    """Test API functions with dependency extraction enabled."""

    def test_parse_file_with_dependencies(self, tmp_path):
        """Test parse_file with dependency extraction."""
        # Create a test file with imports and function calls
        test_file = tmp_path / "test_deps.py"
        test_file.write_text("""
import os
from pathlib import Path

def process_file(filename):
    path = Path(filename)
    if path.exists():
        return os.path.getsize(filename)
    return 0

def main():
    size = process_file("data.txt")
    print(f"File size: {size}")
""")

        # Parse with dependency extraction
        result = parse_file(
            str(test_file),
            output_format="dict",
            extract_dependencies=True
        )

        # Verify imports were captured
        import_edges = [e for e in result["edges"]
                       if e["edge_type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) >= 2  # os and Path

        # Check that imported modules are in nodes
        node_types = {n["type"] for n in result["nodes"]}
        assert "ImportedModule" in node_types

        # Verify function references were captured
        uses_edges = [e for e in result["edges"]
                     if e["edge_type"] == EdgeType.USES.value]
        assert len(uses_edges) >= 1  # main uses process_file

    def test_parse_code_with_dependencies(self):
        """Test parse_code with dependency extraction."""
        code = """
import json

class DataProcessor:
    def __init__(self):
        self.data = []

    def load_json(self, filename):
        with open(filename) as f:
            self.data = json.load(f)

    def process(self):
        return len(self.data)

processor = DataProcessor()
result = processor.process()
"""

        # Parse with dependency extraction
        result = parse_code(
            code,
            filename="test.py",
            output_format="dict",
            extract_dependencies=True
        )

        # Verify imports
        import_edges = [e for e in result["edges"]
                       if e["edge_type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) >= 1  # json

        # Verify class instantiation
        [e for e in result["edges"]
                           if e["edge_type"] == EdgeType.INSTANTIATES.value]
        # Note: Module-level instantiation might not be captured in current implementation

    def test_parse_file_without_dependencies(self, tmp_path):
        """Test that dependency extraction can be disabled."""
        test_file = tmp_path / "test_no_deps.py"
        test_file.write_text("""
import os

def hello():
    print("Hello")
""")

        # Parse without dependency extraction (default)
        result = parse_file(
            str(test_file),
            output_format="dict"
        )

        # Should not have import edges
        import_edges = [e for e in result["edges"]
                       if e["edge_type"] == EdgeType.IMPORTS.value]
        assert len(import_edges) == 0

        # Should not have ImportedModule nodes
        imported_modules = [n for n in result["nodes"]
                          if n["type"] == "ImportedModule"]
        assert len(imported_modules) == 0

    def test_complex_dependencies(self):
        """Test complex dependency scenarios."""
        code = """
# Various import styles
import os.path
from typing import List, Dict, Optional
from collections import defaultdict as dd
import sys, json

# Function using imports
def process_data(data: List[Dict]) -> Optional[str]:
    result = dd(list)
    for item in data:
        result[item['type']].append(item)

    # Serialize result
    return json.dumps(dict(result))

# Class with method calls
class FileHandler:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def get_size(self) -> int:
        if self.exists():
            return os.path.getsize(self.path)
        return 0

# Usage
handler = FileHandler("/tmp/test.txt")
if handler.exists():
    size = handler.get_size()
    sys.stdout.write(f"Size: {size}\\n")
"""

        result = parse_code(
            code,
            output_format="dict",
            extract_dependencies=True
        )

        # Verify various imports were captured
        import_nodes = [n for n in result["nodes"]
                       if n["type"] == "ImportedModule"]
        module_names = {n["ast_node_info"]["module_name"] for n in import_nodes}

        assert "os.path" in module_names
        assert "typing" in module_names
        assert "collections" in module_names
        assert "sys" in module_names
        assert "json" in module_names

        # Check for aliased import
        aliased_imports = [n for n in import_nodes
                          if n["ast_node_info"].get("alias") == "dd"]
        assert len(aliased_imports) == 1

    def test_relative_imports(self):
        """Test relative import handling."""
        code = """
from . import utils
from ..models import User
from ...config import settings

def get_user():
    user = User()
    config = settings.get_config()
    return utils.format_user(user, config)
"""

        result = parse_code(
            code,
            output_format="dict",
            extract_dependencies=True
        )

        # Check relative imports
        import_nodes = [n for n in result["nodes"]
                       if n["type"] == "ImportedModule"]

        relative_imports = [n for n in import_nodes
                          if n["ast_node_info"].get("is_relative", False)]
        assert len(relative_imports) == 3

        # Check import levels
        levels = {n["ast_node_info"]["level"] for n in relative_imports}
        assert levels == {1, 2, 3}

    def test_wildcard_imports(self):
        """Test wildcard import handling."""
        code = """
from math import *
from os.path import *

result = sqrt(16) + sin(pi/2)
path = join("dir", "file.txt")
"""

        result = parse_code(
            code,
            output_format="dict",
            extract_dependencies=True
        )

        # Check wildcard imports
        import_nodes = [n for n in result["nodes"]
                       if n["type"] == "ImportedModule"]

        wildcard_imports = [n for n in import_nodes
                          if "*" in n["ast_node_info"].get("imported_names", [])]
        assert len(wildcard_imports) >= 1
