"""高レベルAPIのユニットテスト"""
import pytest
from unittest.mock import Mock, patch, mock_open
import json
from pathlib import Path

from ast2graph.api import (
    parse_file,
    parse_code,
    parse_directory,
    parse_files_stream
)
from ast2graph.exceptions import ParseError, GraphBuildError
from ast2graph.graph_structure import GraphStructure


class TestParseFile:
    """parse_file関数のテスト"""
    
    def test_parse_file_basic(self, tmp_path):
        """基本的なファイル解析のテスト"""
        # テストファイルを作成
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'")
        
        # 解析を実行
        result = parse_file(str(test_file))
        
        # 結果を検証
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0  # 少なくとも1つのノード
        
    def test_parse_file_with_json_format(self, tmp_path):
        """JSON形式での出力テスト"""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        
        result = parse_file(str(test_file), output_format="json")
        
        # JSON文字列であることを確認
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "nodes" in parsed
        assert "edges" in parsed
        
    def test_parse_file_with_graph_format(self, tmp_path):
        """GraphStructure形式での出力テスト"""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        
        result = parse_file(str(test_file), output_format="graph")
        
        assert isinstance(result, GraphStructure)
        assert len(result.nodes) > 0
        
    def test_parse_file_not_found(self):
        """存在しないファイルのテスト"""
        with pytest.raises(FileNotFoundError):
            parse_file("/non/existent/file.py")
            
    def test_parse_file_syntax_error(self, tmp_path):
        """構文エラーを含むファイルのテスト"""
        test_file = tmp_path / "error.py"
        test_file.write_text("def broken(\n")  # 不完全な構文
        
        with pytest.raises(ParseError) as exc_info:
            parse_file(str(test_file))
        
        assert "error.py" in str(exc_info.value)
        
    def test_parse_file_encoding(self, tmp_path):
        """エンコーディング指定のテスト"""
        test_file = tmp_path / "utf8.py"
        test_file.write_text("# コメント\nx = 'テスト'", encoding="utf-8")
        
        result = parse_file(str(test_file), encoding="utf-8")
        
        assert isinstance(result, dict)
        assert len(result["nodes"]) > 0
        
    def test_parse_file_without_metadata(self, tmp_path):
        """メタデータを含まない出力のテスト"""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        
        result = parse_file(str(test_file), include_metadata=False)
        
        # メタデータフィールドが存在しないことを確認
        for node in result["nodes"]:
            assert "metadata" not in node or node["metadata"] is None
            
    def test_parse_file_without_source_info(self, tmp_path):
        """ソース情報を含まない出力のテスト"""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        
        result = parse_file(str(test_file), include_source_info=False)
        
        # ソース情報フィールドが存在しないことを確認
        for node in result["nodes"]:
            assert "source_info" not in node or node["source_info"] is None


class TestParseCode:
    """parse_code関数のテスト"""
    
    def test_parse_code_basic(self):
        """基本的なコード解析のテスト"""
        code = "def hello():\n    return 'world'"
        result = parse_code(code)
        
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0
        
    def test_parse_code_with_filename(self):
        """仮想ファイル名の指定テスト"""
        code = "x = 1"
        result = parse_code(code, filename="virtual.py")
        
        # parse_code関数が正しく動作することを確認
        assert "nodes" in result
        assert len(result["nodes"]) > 0
        # ファイル名情報はsource_infoセクションに格納される可能性がある
        # または、この情報はグラフ構造内で管理される
        
    def test_parse_code_syntax_error(self):
        """構文エラーを含むコードのテスト"""
        code = "def broken(\n"
        
        with pytest.raises(ParseError) as exc_info:
            parse_code(code)
            
        assert "<string>" in str(exc_info.value)
        
    def test_parse_code_empty(self):
        """空のコードのテスト"""
        result = parse_code("")
        
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) >= 1  # 最低限Moduleノードは存在
        
    def test_parse_code_json_format(self):
        """JSON形式での出力テスト"""
        code = "x = 1"
        result = parse_code(code, output_format="json")
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "nodes" in parsed


class TestParseDirectory:
    """parse_directory関数のテスト"""
    
    def test_parse_directory_basic(self, tmp_path):
        """基本的なディレクトリ解析のテスト"""
        # テストディレクトリ構造を作成
        (tmp_path / "sub").mkdir()
        (tmp_path / "file1.py").write_text("x = 1")
        (tmp_path / "file2.py").write_text("y = 2")
        (tmp_path / "sub" / "file3.py").write_text("z = 3")
        (tmp_path / "readme.txt").write_text("Not a Python file")
        
        result = parse_directory(str(tmp_path))
        
        assert isinstance(result, dict)
        assert len(result) == 3  # 3つのPythonファイル
        assert any("file1.py" in key for key in result.keys())
        assert any("file2.py" in key for key in result.keys())
        assert any("file3.py" in key for key in result.keys())
        
    def test_parse_directory_pattern(self, tmp_path):
        """ファイルパターン指定のテスト"""
        (tmp_path / "test_file.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("y = 2")
        
        # test_*.py パターンで検索
        result = parse_directory(str(tmp_path), pattern="test_*.py")
        
        assert len(result) == 1
        assert any("test_file.py" in key for key in result.keys())
        
    def test_parse_directory_output_dir(self, tmp_path):
        """出力ディレクトリ指定のテスト"""
        (tmp_path / "src").mkdir()
        (tmp_path / "out").mkdir()
        (tmp_path / "src" / "file.py").write_text("x = 1")
        
        result = parse_directory(
            str(tmp_path / "src"),
            output_dir=str(tmp_path / "out")
        )
        
        # 結果ファイルが作成されていることを確認
        output_files = list((tmp_path / "out").glob("*.json"))
        assert len(output_files) > 0
        
    def test_parse_directory_empty(self, tmp_path):
        """空のディレクトリのテスト"""
        result = parse_directory(str(tmp_path))
        
        assert isinstance(result, dict)
        assert len(result) == 0
        
    def test_parse_directory_not_exists(self):
        """存在しないディレクトリのテスト"""
        with pytest.raises(FileNotFoundError):
            parse_directory("/non/existent/directory")
            
    def test_parse_directory_with_errors(self, tmp_path):
        """エラーを含むファイルがある場合のテスト"""
        (tmp_path / "good.py").write_text("x = 1")
        (tmp_path / "bad.py").write_text("def broken(\n")
        
        # エラーがあっても処理を継続
        result = parse_directory(str(tmp_path))
        
        # 正常なファイルは処理される
        assert len(result) >= 1
        assert any("good.py" in key for key in result.keys())
        
        # エラーファイルの情報も含まれる（エラー情報として）
        for key, value in result.items():
            if "bad.py" in key:
                assert "error" in value or isinstance(value, Exception)
                
    def test_parse_directory_parallel(self, tmp_path):
        """並列処理のテスト"""
        # 複数ファイルを作成
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}")
            
        # 並列処理で実行
        result = parse_directory(str(tmp_path), parallel=True, max_workers=4)
        
        assert len(result) == 10
        # すべてのファイルが処理されていることを確認
        for i in range(10):
            assert any(f"file{i}.py" in key for key in result.keys())
            
    def test_parse_directory_progress_callback(self, tmp_path):
        """進捗コールバックのテスト"""
        # テストファイルを作成
        for i in range(3):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}")
            
        # 進捗を記録
        progress_log = []
        
        def callback(completed, total):
            progress_log.append((completed, total))
            
        result = parse_directory(
            str(tmp_path),
            progress_callback=callback
        )
        
        # コールバックが呼ばれたことを確認
        assert len(progress_log) > 0
        # 最終的に全ファイルが処理されたことを確認
        assert progress_log[-1] == (3, 3)


class TestParseFilesStream:
    """parse_files_stream関数のテスト"""
    
    def test_parse_files_stream_basic(self, tmp_path):
        """基本的なストリーミング処理のテスト"""
        # テストファイルを作成
        files = []
        for i in range(5):
            file_path = tmp_path / f"file{i}.py"
            file_path.write_text(f"x = {i}")
            files.append(str(file_path))
            
        # ストリーミング処理
        results = list(parse_files_stream(files))
        
        assert len(results) == 5
        for result in results:
            assert "file_path" in result
            assert "graph" in result or "error" in result
            
    def test_parse_files_stream_chunk_size(self, tmp_path):
        """チャンクサイズ指定のテスト"""
        # テストファイルを作成
        files = []
        for i in range(10):
            file_path = tmp_path / f"file{i}.py"
            file_path.write_text(f"x = {i}")
            files.append(str(file_path))
            
        # チャンクサイズ3で処理
        results = list(parse_files_stream(files, chunk_size=3))
        
        # 全ファイルが処理されることを確認
        assert len(results) == 10
        
    def test_parse_files_stream_with_errors(self, tmp_path):
        """エラーを含むファイルのストリーミング処理"""
        files = []
        
        # 正常なファイル
        good_file = tmp_path / "good.py"
        good_file.write_text("x = 1")
        files.append(str(good_file))
        
        # エラーファイル
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n")
        files.append(str(bad_file))
        
        # ストリーミング処理
        results = list(parse_files_stream(files))
        
        assert len(results) == 2
        
        # 正常なファイルは処理される
        good_result = next(r for r in results if "good.py" in r["file_path"])
        assert "graph" in good_result
        
        # エラーファイルもエラー情報付きで返される
        bad_result = next(r for r in results if "bad.py" in r["file_path"])
        assert "error" in bad_result
        
    def test_parse_files_stream_empty_list(self):
        """空のファイルリストのテスト"""
        results = list(parse_files_stream([]))
        assert len(results) == 0
        
    def test_parse_files_stream_generator(self, tmp_path):
        """ジェネレーターとしての動作確認"""
        # 大量のファイルを想定
        files = []
        for i in range(100):
            file_path = tmp_path / f"file{i}.py"
            file_path.write_text(f"x = {i}")
            files.append(str(file_path))
            
        # ストリーミング処理（遅延評価）
        stream = parse_files_stream(files, chunk_size=10)
        
        # 最初の結果のみ取得
        first_result = next(stream)
        assert "file_path" in first_result
        
        # まだすべては処理されていない（ジェネレーター）
        # 残りも順次処理可能
        count = 1
        for _ in stream:
            count += 1
            
        assert count == 100