"""エラーシナリオの統合テスト.

構文エラー、エンコーディング問題、循環参照、大規模ファイル等の
エラーケースを網羅的にテストする。
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from ast2graph import parse_code, parse_directory, parse_file, parse_files_stream
from ast2graph.exceptions import (
    AST2GraphError,
    ParseError,
    GraphBuildError,
    ValidationError,
)


class TestSyntaxErrorHandling:
    """構文エラーを含むファイルのテスト."""

    def test_simple_syntax_error(self, tmp_path: Path) -> None:
        """単純な構文エラーの処理."""
        # Arrange
        code = '''
def broken_function(
    # Missing closing parenthesis
    return "This won't work"
'''
        file_path = tmp_path / "syntax_error.py"
        file_path.write_text(code)

        # Act & Assert
        with pytest.raises(ParseError) as exc_info:
            parse_file(str(file_path))
        
        assert "syntax" in str(exc_info.value).lower()

    def test_multiple_syntax_errors(self, tmp_path: Path) -> None:
        """複数の構文エラーを含むファイル."""
        # Arrange
        code = '''
def func1(
    pass  # Missing closing parenthesis

class BrokenClass
    def method(self):  # Missing colon after class
        return

if True
    print("Missing colon")  # Missing colon after if
'''
        file_path = tmp_path / "multiple_errors.py"
        file_path.write_text(code)

        # Act & Assert
        with pytest.raises(ParseError):
            parse_file(str(file_path))

    def test_partial_parsing_mode(self, tmp_path: Path) -> None:
        """部分的な解析モード（将来の拡張用）."""
        # Arrange
        code = '''
# Valid part
def valid_function():
    return "This is valid"

# Invalid part
def invalid_function(
    return "This is invalid"

# Another valid part
class ValidClass:
    pass
'''
        # Note: 現在の実装では部分解析はサポートされていないが、
        # 将来の拡張のためのテストケースとして残す
        file_path = tmp_path / "partial.py"
        file_path.write_text(code)

        # Act & Assert
        with pytest.raises(ParseError):
            parse_file(str(file_path))

    def test_indentation_errors(self, tmp_path: Path) -> None:
        """インデントエラーのテスト."""
        # Arrange
        code = '''
def function_with_bad_indent():
return "Bad indent"  # No indentation

    def another_function():
        if True:
    print("Inconsistent indent")  # Wrong indentation level
'''
        file_path = tmp_path / "indent_error.py"
        file_path.write_text(code)

        # Act & Assert
        with pytest.raises(ParseError) as exc_info:
            parse_file(str(file_path))
        
        error_msg = str(exc_info.value).lower()
        assert "indent" in error_msg or "syntax" in error_msg


class TestEncodingProblems:
    """エンコーディング問題のテスト."""

    def test_utf8_bom_file(self, tmp_path: Path) -> None:
        """UTF-8 BOM付きファイルの処理."""
        # Arrange
        code = '''def hello():
    return "こんにちは"
'''
        file_path = tmp_path / "utf8_bom.py"
        # UTF-8 BOMを付けて書き込み
        file_path.write_bytes(b'\xef\xbb\xbf' + code.encode('utf-8'))

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        assert len(func_nodes) == 1
        assert func_nodes[0]["properties"]["name"] == "hello"

    def test_latin1_encoding(self, tmp_path: Path) -> None:
        """Latin-1エンコーディングのファイル."""
        # Arrange
        code = '''# -*- coding: latin-1 -*-
def greet():
    return "Café"  # Latin-1 character
'''
        file_path = tmp_path / "latin1.py"
        file_path.write_bytes(code.encode('latin-1'))

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        assert len(func_nodes) == 1

    def test_invalid_encoding(self, tmp_path: Path) -> None:
        """無効なエンコーディングのファイル."""
        # Arrange
        file_path = tmp_path / "invalid_encoding.py"
        # 無効なバイトシーケンスを書き込み
        file_path.write_bytes(b'\xff\xfe\x00\x00Invalid UTF content')

        # Act & Assert
        with pytest.raises(ParseError) as exc_info:
            parse_file(str(file_path))
        
        assert "decod" in str(exc_info.value).lower() or "encod" in str(exc_info.value).lower()

    def test_mixed_encoding_in_directory(self, tmp_path: Path) -> None:
        """異なるエンコーディングのファイルが混在するディレクトリ."""
        # Arrange
        project = tmp_path / "mixed_encoding"
        project.mkdir()
        
        # UTF-8ファイル
        (project / "utf8.py").write_text('def func(): return "UTF-8"', encoding='utf-8')
        
        # Latin-1ファイル
        latin1_code = '# -*- coding: latin-1 -*-\ndef func(): return "Café"'
        (project / "latin1.py").write_bytes(latin1_code.encode('latin-1'))
        
        # ASCII ファイル
        (project / "ascii.py").write_text('def func(): return "ASCII"')

        # Act
        results = parse_directory(str(project))

        # Assert
        assert len(results) == 3
        # エラーが発生したファイルもresultsに含まれる（エラー情報付き）
        for file_path, result in results.items():
            assert isinstance(result, dict)
            # 成功したファイルにはnodesが含まれる
            if "error" not in result:
                assert "nodes" in result


class TestCircularReferences:
    """循環参照のテスト."""

    def test_simple_circular_import(self, tmp_path: Path) -> None:
        """単純な循環インポート."""
        # Arrange
        project = tmp_path / "circular"
        project.mkdir()
        
        # a.py imports b
        (project / "a.py").write_text('''
from . import b

def func_a():
    return b.func_b() + " from a"
''')
        
        # b.py imports a
        (project / "b.py").write_text('''
from . import a

def func_b():
    return "b"

def use_a():
    return a.func_a()
''')

        # Act - 循環インポート自体はPythonで有効なので、エラーにはならない
        results = parse_directory(str(project), include_dependencies=True)

        # Assert
        assert len(results) == 2
        
        # 両方のファイルが正常に解析される
        for file_path in results:
            assert results[file_path] is not None
            assert "nodes" in results[file_path]
            assert "edges" in results[file_path]

    def test_complex_circular_dependency_chain(self, tmp_path: Path) -> None:
        """複雑な循環依存チェーン."""
        # Arrange
        project = tmp_path / "complex_circular"
        project.mkdir()
        
        # A -> B -> C -> D -> A の循環
        modules = {
            "module_a.py": '''
from . import module_b

class A:
    def use_b(self):
        return module_b.B()
''',
            "module_b.py": '''
from . import module_c

class B:
    def use_c(self):
        return module_c.C()
''',
            "module_c.py": '''
from . import module_d

class C:
    def use_d(self):
        return module_d.D()
''',
            "module_d.py": '''
from . import module_a

class D:
    def use_a(self):
        # This creates circular dependency
        return module_a.A()
'''
        }
        
        for filename, content in modules.items():
            (project / filename).write_text(content)

        # Act
        results = parse_directory(str(project), include_dependencies=True)

        # Assert
        assert len(results) == 4
        
        # 各モジュールのインポート関係を確認
        for file_path, result in results.items():
            edges = result["edges"]
            import_edges = [e for e in edges if e["type"] == "IMPORTS"]
            assert len(import_edges) >= 1  # 各モジュールが次のモジュールをインポート

    def test_self_referential_import(self, tmp_path: Path) -> None:
        """自己参照的なインポート."""
        # Arrange
        code = '''
# This is unusual but valid Python
import __main__ as myself

def recursive_function(n):
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)

class SelfReferential:
    def get_class(self):
        return SelfReferential
'''
        file_path = tmp_path / "self_ref.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path), include_dependencies=True)

        # Assert
        assert result is not None
        nodes = result["nodes"]
        
        # 再帰関数の確認
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        assert any(n["properties"]["name"] == "recursive_function" for n in func_nodes)


class TestLargeFileHandling:
    """大規模ファイルの処理テスト."""

    def test_file_with_many_functions(self, tmp_path: Path) -> None:
        """多数の関数を含むファイル."""
        # Arrange
        lines = ['"""Large file with many functions."""']
        for i in range(500):  # 500個の関数
            lines.extend([
                f"\ndef function_{i}(x, y):",
                f'    """Function {i} docstring."""',
                f"    result = x + y + {i}",
                f"    return result * {i + 1}",
            ])
        
        code = "\n".join(lines)
        file_path = tmp_path / "many_functions.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        assert len(func_nodes) == 500
        
        # メタデータの確認
        metadata = result["metadata"]
        assert metadata["total_nodes"] > 500  # 関数以外のノードも含む
        assert metadata["source_file"] == str(file_path)

    def test_file_with_deep_nesting(self, tmp_path: Path) -> None:
        """深いネスト構造を持つファイル."""
        # Arrange
        lines = ['"""Deeply nested structure."""']
        
        # 深くネストした条件文
        indent = ""
        for i in range(20):  # 20レベルのネスト
            lines.append(f"{indent}if True:  # Level {i}")
            indent += "    "
        lines.append(f'{indent}result = "Deep!"')
        
        # 深くネストしたクラス定義
        lines.append("\n\nclass OuterClass:")
        indent = "    "
        for i in range(10):
            lines.append(f"{indent}class InnerClass{i}:")
            indent += "    "
        lines.append(f"{indent}pass")
        
        code = "\n".join(lines)
        file_path = tmp_path / "deep_nesting.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        
        # ネストした構造が正しく解析されているか
        if_nodes = [n for n in nodes if n["type"] == "If"]
        assert len(if_nodes) >= 20
        
        class_nodes = [n for n in nodes if n["type"] == "ClassDef"]
        assert len(class_nodes) >= 10

    def test_file_with_large_data_structures(self, tmp_path: Path) -> None:
        """大きなデータ構造を含むファイル."""
        # Arrange
        lines = ['"""File with large data structures."""']
        
        # 大きなリスト
        lines.append("\nlarge_list = [")
        for i in range(1000):
            lines.append(f"    {i},")
        lines.append("]")
        
        # 大きな辞書
        lines.append("\nlarge_dict = {")
        for i in range(500):
            lines.append(f'    "key_{i}": {i * 2},')
        lines.append("}")
        
        # 大きなタプル
        lines.append("\nlarge_tuple = (")
        lines.extend([f"    {i}," for i in range(500)])
        lines.append(")")
        
        code = "\n".join(lines)
        file_path = tmp_path / "large_data.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        
        # 代入文の確認
        assign_nodes = [n for n in nodes if n["type"] == "Assign"]
        assert len(assign_nodes) >= 3  # large_list, large_dict, large_tuple
        
        # リスト、辞書、タプルノードの存在確認
        list_nodes = [n for n in nodes if n["type"] == "List"]
        dict_nodes = [n for n in nodes if n["type"] == "Dict"]
        tuple_nodes = [n for n in nodes if n["type"] == "Tuple"]
        
        assert len(list_nodes) >= 1
        assert len(dict_nodes) >= 1
        assert len(tuple_nodes) >= 1


class TestErrorRecoveryStrategies:
    """エラーリカバリー戦略のテスト."""

    def test_skip_errors_in_directory_parsing(self, tmp_path: Path) -> None:
        """ディレクトリ解析時のエラースキップ."""
        # Arrange
        project = tmp_path / "mixed_files"
        project.mkdir()
        
        # 正常なファイル
        (project / "good1.py").write_text('def good(): return "OK"')
        (project / "good2.py").write_text('class Good: pass')
        
        # エラーを含むファイル
        (project / "bad_syntax.py").write_text('def bad syntax():')
        (project / "bad_encoding.py").write_bytes(b'\xff\xfe\x00\x00')
        
        # 正常なファイル
        (project / "good3.py").write_text('x = 42')

        # Act
        results = parse_directory(str(project), skip_errors=True)

        # Assert
        assert len(results) == 5  # 全ファイル分の結果
        
        # 正常なファイルは解析成功
        assert results[str(project / "good1.py")] is not None
        assert results[str(project / "good2.py")] is not None
        assert results[str(project / "good3.py")] is not None
        
        # エラーファイルはNone
        assert results[str(project / "bad_syntax.py")] is None
        assert results[str(project / "bad_encoding.py")] is None

    def test_streaming_with_mixed_errors(self, tmp_path: Path) -> None:
        """ストリーミング処理での混在エラー処理."""
        # Arrange
        files = []
        
        # 様々な種類のファイルを作成
        for i in range(10):
            if i % 3 == 0:
                # 構文エラー
                file_path = tmp_path / f"syntax_error_{i}.py"
                file_path.write_text(f"def invalid syntax {i}")
            elif i % 3 == 1:
                # 正常なファイル
                file_path = tmp_path / f"valid_{i}.py"
                file_path.write_text(f"def func_{i}(): return {i}")
            else:
                # エンコーディングエラー
                file_path = tmp_path / f"encoding_error_{i}.py"
                file_path.write_bytes(b'\xff\xfe' + f"invalid {i}".encode('ascii', errors='ignore'))
            
            files.append(str(file_path))

        # Act
        results = list(parse_files_stream(files, skip_errors=True))

        # Assert
        assert len(results) == 10
        
        # 正常なファイルの数を確認
        valid_results = [r for _, r in results if r is not None]
        assert len(valid_results) == 3  # i % 3 == 1 のケース
        
        # エラーファイルの数を確認
        error_results = [r for _, r in results if r is None]
        assert len(error_results) == 7  # 残りのケース

    def test_graceful_degradation(self, tmp_path: Path) -> None:
        """段階的な機能低下のテスト."""
        # Arrange
        code_with_advanced_features = '''
# Python 3.10+ features that might not parse in older versions
def func_with_match(value):
    match value:
        case 0:
            return "zero"
        case n if n > 0:
            return "positive"
        case _:
            return "negative"

# Type annotations
from typing import TypeAlias
Number: TypeAlias = int | float

# Walrus operator
if (n := len([1, 2, 3])) > 2:
    print(f"Length is {n}")
'''
        file_path = tmp_path / "advanced.py"
        file_path.write_text(code_with_advanced_features)

        # Act
        try:
            result = parse_file(str(file_path))
            # Python 3.10+で実行された場合
            assert result is not None
            nodes = result["nodes"]
            assert len(nodes) > 0
        except ParseError:
            # 古いPythonバージョンでは構文エラーになる可能性
            # これは期待される動作
            pass

    def test_permission_errors(self, tmp_path: Path) -> None:
        """パーミッションエラーのテスト."""
        # Arrange
        file_path = tmp_path / "readonly.py"
        file_path.write_text('def test(): return "OK"')
        
        # ファイルを読み取り専用に設定
        os.chmod(file_path, 0o444)
        
        try:
            # Act - 読み取りは成功するはず
            result = parse_file(str(file_path))
            
            # Assert
            assert result is not None
            nodes = result["nodes"]
            func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
            assert len(func_nodes) == 1
            
        finally:
            # クリーンアップ：権限を戻す
            os.chmod(file_path, 0o644)

    def test_non_existent_file(self) -> None:
        """存在しないファイルの処理."""
        # Arrange
        non_existent = "/path/to/non/existent/file.py"
        
        # Act & Assert
        with pytest.raises(AST2GraphError):
            parse_file(non_existent)


class TestEdgeCases:
    """エッジケースのテスト."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """空ファイルの処理."""
        # Arrange
        file_path = tmp_path / "empty.py"
        file_path.write_text("")

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["metadata"]["total_nodes"] == 0

    def test_only_comments_and_docstrings(self, tmp_path: Path) -> None:
        """コメントとドキュメント文字列のみのファイル."""
        # Arrange
        code = '''
"""
This is a module docstring.
It contains multiple lines.
"""

# This is a comment
# Another comment

"""
This is just a string literal, not a docstring.
"""

# More comments
'''
        file_path = tmp_path / "comments_only.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        
        # モジュールレベルの文字列リテラルは Expr ノードとして解析される
        expr_nodes = [n for n in nodes if n["type"] == "Expr"]
        assert len(expr_nodes) >= 1

    def test_extremely_long_lines(self, tmp_path: Path) -> None:
        """非常に長い行を含むファイル."""
        # Arrange
        # 10000文字の変数名
        long_var_name = "x" * 10000
        code = f'{long_var_name} = 42\n'
        
        # 1000個の要素を持つリスト（1行）
        code += 'long_list = [' + ', '.join(str(i) for i in range(1000)) + ']\n'
        
        file_path = tmp_path / "long_lines.py"
        file_path.write_text(code)

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        assign_nodes = [n for n in nodes if n["type"] == "Assign"]
        assert len(assign_nodes) >= 2

    def test_unicode_identifiers(self, tmp_path: Path) -> None:
        """Unicode識別子のテスト."""
        # Arrange
        code = '''
# Python 3 allows unicode identifiers
def 你好():
    return "Hello in Chinese"

class 日本語クラス:
    def メソッド(self):
        return "Japanese method"

π = 3.14159
Δx = 0.001
'''
        file_path = tmp_path / "unicode.py"
        file_path.write_text(code, encoding='utf-8')

        # Act
        result = parse_file(str(file_path))

        # Assert
        assert result is not None
        nodes = result["nodes"]
        
        # Unicode関数名
        func_nodes = [n for n in nodes if n["type"] == "FunctionDef"]
        assert any(n["properties"]["name"] == "你好" for n in func_nodes)
        
        # Unicodeクラス名
        class_nodes = [n for n in nodes if n["type"] == "ClassDef"]
        assert any(n["properties"]["name"] == "日本語クラス" for n in class_nodes)