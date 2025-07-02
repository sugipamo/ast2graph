"""ast2graphの高レベルAPI

エンドユーザー向けのシンプルなインターフェースを提供します。
"""
import json
from pathlib import Path
from typing import Union, Dict, Any, List, Iterator, Optional, Callable
import concurrent.futures
import glob
import uuid

from .parser import ASTParser
from .graph_builder import GraphBuilder
from .graph_exporter import GraphExporter
from .graph_structure import GraphStructure, SourceInfo
from .exceptions import ParseError, GraphBuildError, FileReadError
from .dependency_extractor import DependencyExtractor


def parse_file(
    file_path: str,
    output_format: str = "dict",
    include_metadata: bool = True,
    include_source_info: bool = True,
    encoding: str = "utf-8",
    extract_dependencies: bool = False
) -> Union[Dict[str, Any], str, GraphStructure]:
    """単一のPythonファイルを解析してグラフに変換する。
    
    Args:
        file_path: 解析対象ファイルのパス
        output_format: 出力形式 ("dict", "json", "graph")
        include_metadata: メタデータを含めるか
        include_source_info: ソース情報を含めるか
        encoding: ファイルエンコーディング
        extract_dependencies: 依存関係を抽出するか
        
    Returns:
        指定された形式でのグラフデータ
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ParseError: 構文解析エラー
        GraphBuildError: グラフ構築エラー
    """
    # パーサーを使ってファイルを直接解析
    parser = ASTParser()
    parse_result = parser.parse_file(file_path)
    
    # エラーチェック
    if parse_result.error:
        if isinstance(parse_result.error, FileReadError):
            raise parse_result.error.original_error
        elif isinstance(parse_result.error, ParseError):
            raise parse_result.error
        else:
            raise ParseError(f"Failed to parse {file_path}: {str(parse_result.error)}")
            
    ast_tree = parse_result.ast
        
    # グラフ構築
    source_info = SourceInfo(
        source_id=str(uuid.uuid4()),
        file_path=file_path,
        file_hash=parse_result.file_hash or parse_result.source_hash,
        parsed_at=parse_result.parsed_at,
        encoding=parse_result.encoding,
        line_count=parse_result.line_count
    )
    
    builder = GraphBuilder(ast_tree, source_info)
    try:
        graph = builder.build_graph()
    except Exception as e:
        raise GraphBuildError(f"Failed to build graph for {file_path}: {str(e)}")
    
    # 依存関係抽出（オプション）
    if extract_dependencies:
        extractor = DependencyExtractor()
        # モジュールノードを取得（通常は最初のノード）
        module_node_id = None
        for node_id, node in graph.nodes.items():
            if node.node_type == "Module":
                module_node_id = node_id
                break
        extractor.extract_dependencies(ast_tree, graph, module_node_id)
        
    # エクスポート
    exporter = GraphExporter(graph)
    
    # メタデータとソース情報の設定
    if not include_metadata:
        # メタデータを削除
        for node in graph.nodes.values():
            node.metadata = None
    if not include_source_info:
        # ソース情報を削除
        for node in graph.nodes.values():
            if hasattr(node, 'source_info'):
                node.source_info = None
                
    # 出力形式に応じて変換
    if output_format == "graph":
        return graph
    elif output_format == "json":
        return exporter.export_to_json()
    else:  # dict
        return exporter.export_to_dict(include_metadata=include_metadata, include_source_info=include_source_info)


def parse_code(
    source_code: str,
    filename: str = "<string>",
    output_format: str = "dict",
    include_metadata: bool = True,
    extract_dependencies: bool = False
) -> Union[Dict[str, Any], str, GraphStructure]:
    """文字列として提供されたPythonコードを解析する。
    
    Args:
        source_code: 解析対象のPythonコード
        filename: 仮想ファイル名（エラー表示用）
        output_format: 出力形式 ("dict", "json", "graph")
        include_metadata: メタデータを含めるか
        extract_dependencies: 依存関係を抽出するか
        
    Returns:
        指定された形式でのグラフデータ
        
    Raises:
        ParseError: 構文解析エラー
        GraphBuildError: グラフ構築エラー
    """
    # パース処理
    parser = ASTParser()
    parse_result = parser.parse_code(source_code, filename)
    
    # エラーチェック
    if parse_result.error:
        raise ParseError(f"Parse error in {filename}: {str(parse_result.error)}")
        
    ast_tree = parse_result.ast
        
    # グラフ構築
    source_info = SourceInfo(
        source_id=str(uuid.uuid4()),
        file_path=filename,
        file_hash=parse_result.source_hash,
        parsed_at=parse_result.parsed_at,
        encoding=parse_result.encoding,
        line_count=parse_result.line_count
    )
    
    builder = GraphBuilder(ast_tree, source_info)
    try:
        graph = builder.build_graph()
    except Exception as e:
        raise GraphBuildError(f"Failed to build graph: {str(e)}")
    
    # 依存関係抽出（オプション）
    if extract_dependencies:
        extractor = DependencyExtractor()
        # モジュールノードを取得（通常は最初のノード）
        module_node_id = None
        for node_id, node in graph.nodes.items():
            if node.node_type == "Module":
                module_node_id = node_id
                break
        extractor.extract_dependencies(ast_tree, graph, module_node_id)
        
    # エクスポート
    exporter = GraphExporter(graph)
    
    # メタデータの設定
    if not include_metadata:
        for node in graph.nodes.values():
            node.metadata = None
            
    # 出力形式に応じて変換
    if output_format == "graph":
        return graph
    elif output_format == "json":
        return exporter.export_to_json()
    else:  # dict
        return exporter.export_to_dict(include_metadata=include_metadata)


def parse_directory(
    directory_path: str,
    pattern: str = "**/*.py",
    output_dir: Optional[str] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Union[GraphStructure, Dict[str, Any]]]:
    """ディレクトリ内のすべてのPythonファイルを解析する。
    
    Args:
        directory_path: 解析対象ディレクトリ
        pattern: ファイルパターン（glob形式）
        output_dir: 結果の出力先ディレクトリ
        parallel: 並列処理を使用するか
        max_workers: 最大ワーカー数
        progress_callback: 進捗通知用コールバック
        
    Returns:
        ファイルパスをキーとする解析結果の辞書
        
    Raises:
        FileNotFoundError: ディレクトリが存在しない場合
    """
    # ディレクトリの存在確認
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
        
    # ファイルリストの取得
    file_paths = []
    for file_path in glob.glob(str(dir_path / pattern), recursive=True):
        if Path(file_path).is_file():
            file_paths.append(file_path)
            
    # 結果の格納用辞書
    results = {}
    completed = 0
    total = len(file_paths)
    
    # 進捗通知
    if progress_callback:
        progress_callback(completed, total)
        
    # 出力ディレクトリの準備
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
    def process_single_file(file_path: str) -> tuple[str, Union[Dict[str, Any], Exception]]:
        """単一ファイルを処理"""
        try:
            result = parse_file(file_path)
            
            # 出力ディレクトリが指定されている場合はファイルに保存
            if output_dir:
                output_file = output_path / f"{Path(file_path).stem}_graph.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                    
            return file_path, result
        except Exception as e:
            # エラーも結果として返す
            return file_path, {"error": str(e), "type": type(e).__name__}
            
    if parallel and len(file_paths) > 1:
        # 並列処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_file, file_path): file_path
                for file_path in file_paths
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, result = future.result()
                results[file_path] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total)
    else:
        # 逐次処理
        for file_path in file_paths:
            file_path, result = process_single_file(file_path)
            results[file_path] = result
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total)
                
    return results


def parse_files_stream(
    file_paths: List[str],
    chunk_size: int = 50
) -> Iterator[Dict[str, Any]]:
    """大規模ファイルセットのストリーミング処理。
    
    ファイルを順次処理し、結果を逐次的に返します。
    メモリ効率的な処理が可能です。
    
    Args:
        file_paths: 解析対象ファイルパスのリスト
        chunk_size: 一度に処理するファイル数（並列処理時）
        
    Yields:
        各ファイルの処理結果を含む辞書
        {
            "file_path": str,
            "graph": Dict[str, Any] または "error": str
        }
    """
    # チャンクごとに処理
    for i in range(0, len(file_paths), chunk_size):
        chunk = file_paths[i:i + chunk_size]
        
        # 並列処理でチャンクを処理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            for file_path in chunk:
                future = executor.submit(_parse_file_for_stream, file_path)
                futures.append((file_path, future))
                
            # 結果を順次yield
            for file_path, future in futures:
                try:
                    result = future.result()
                    yield {
                        "file_path": file_path,
                        "graph": result
                    }
                except Exception as e:
                    yield {
                        "file_path": file_path,
                        "error": str(e)
                    }


def _parse_file_for_stream(file_path: str) -> Dict[str, Any]:
    """ストリーミング処理用の内部関数"""
    return parse_file(file_path)