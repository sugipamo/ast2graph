"""Tests for AST parser functionality."""
import ast as ast_module
import tempfile
from pathlib import Path
from unittest import TestCase

from ast2graph.exceptions import FileReadError, ParseError
from ast2graph.parser import ASTParser, ParseResult


class TestASTParser(TestCase):
    """Test cases for ASTParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = ASTParser()

    def test_parse_simple_code(self):
        """Test parsing simple Python code."""
        code = "x = 1"
        result = self.parser.parse_code(code)

        self.assertIsInstance(result, ParseResult)
        self.assertIsInstance(result.ast, ast_module.Module)
        self.assertIsNone(result.error)
        self.assertEqual(result.source_code, code)
        self.assertEqual(result.line_count, 1)

    def test_parse_function_definition(self):
        """Test parsing function definition."""
        code = """
def hello(name):
    return f"Hello, {name}!"
"""
        result = self.parser.parse_code(code)

        self.assertIsInstance(result, ParseResult)
        self.assertIsInstance(result.ast, ast_module.Module)
        self.assertEqual(len(result.ast.body), 1)
        self.assertIsInstance(result.ast.body[0], ast_module.FunctionDef)

    def test_parse_class_definition(self):
        """Test parsing class definition."""
        code = """
class MyClass:
    def __init__(self, value):
        self.value = value
"""
        result = self.parser.parse_code(code)

        self.assertIsInstance(result, ParseResult)
        self.assertIsInstance(result.ast.body[0], ast_module.ClassDef)

    def test_parse_syntax_error(self):
        """Test handling of syntax errors."""
        code = "def invalid syntax:"
        result = self.parser.parse_code(code)

        self.assertIsNotNone(result.error)
        self.assertIsInstance(result.error, ParseError)
        self.assertIn("expected", str(result.error))  # Syntax error message contains 'expected'
        self.assertIsNone(result.ast)

    def test_parse_empty_code(self):
        """Test parsing empty code."""
        code = ""
        result = self.parser.parse_code(code)

        self.assertIsInstance(result.ast, ast_module.Module)
        self.assertEqual(len(result.ast.body), 0)

    def test_parse_file_success(self):
        """Test parsing a file successfully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("x = 42\nprint(x)")
            temp_path = f.name

        try:
            result = self.parser.parse_file(temp_path)

            self.assertIsInstance(result, ParseResult)
            self.assertIsInstance(result.ast, ast_module.Module)
            self.assertEqual(result.file_path, temp_path)
            self.assertEqual(result.line_count, 2)
            self.assertIsNotNone(result.file_hash)
        finally:
            Path(temp_path).unlink()

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file."""
        result = self.parser.parse_file("/nonexistent/file.py")

        self.assertIsNotNone(result.error)
        self.assertIsInstance(result.error, FileReadError)
        self.assertIsNone(result.ast)

    def test_parse_file_with_encoding(self):
        """Test parsing file with different encoding."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.py', delete=False) as f:
            f.write("# -*- coding: utf-8 -*-\nname = '日本語'")
            temp_path = f.name

        try:
            result = self.parser.parse_file(temp_path)

            self.assertIsInstance(result.ast, ast_module.Module)
            self.assertEqual(result.encoding, 'utf-8')
        finally:
            Path(temp_path).unlink()

    def test_parse_result_metadata(self):
        """Test ParseResult metadata."""
        code = """
import os
import sys

def main():
    pass
"""
        result = self.parser.parse_code(code)

        self.assertEqual(result.line_count, 7)
        self.assertGreater(len(result.source_hash), 0)
        self.assertIsNotNone(result.parsed_at)

    def test_parse_complex_syntax(self):
        """Test parsing complex Python syntax."""
        code = """
from typing import List, Dict, Optional

class DataProcessor:
    def __init__(self, data: List[Dict[str, any]]):
        self.data = data

    async def process(self) -> Optional[Dict]:
        for item in self.data:
            if result := self._validate(item):
                yield result

    @staticmethod
    def _validate(item: Dict) -> bool:
        return 'id' in item
"""
        result = self.parser.parse_code(code)

        self.assertIsInstance(result.ast, ast_module.Module)
        self.assertIsNone(result.error)

    def test_parse_with_syntax_error_details(self):
        """Test syntax error details are captured."""
        code = """
def func():
    x = 1
        y = 2  # Indentation error
"""
        result = self.parser.parse_code(code)

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.line_no, 4)
        self.assertGreater(result.error.col_offset, 0)

    def test_parse_file_permission_denied(self):
        """Test handling permission denied error."""
        # Create a file and remove read permissions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("x = 1")
            temp_path = f.name

        try:
            # Remove read permissions
            Path(temp_path).chmod(0o000)

            result = self.parser.parse_file(temp_path)

            self.assertIsNotNone(result.error)
            self.assertIsInstance(result.error, FileReadError)
        finally:
            # Restore permissions and delete
            Path(temp_path).chmod(0o644)
            Path(temp_path).unlink()

    def test_detect_encoding_from_declaration(self):
        """Test encoding detection from coding declaration."""
        code = "# -*- coding: latin-1 -*-\nx = 'test'"
        result = self.parser.parse_code(code)

        # When parsing from string, encoding should be utf-8 (default)
        self.assertEqual(result.encoding, 'utf-8')
