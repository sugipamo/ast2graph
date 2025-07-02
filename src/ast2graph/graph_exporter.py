"""グラフ構造をJSON形式にエクスポートする機能を提供するモジュール。"""

import json
from typing import Any, Dict, IO, Optional, List
from datetime import datetime
import uuid

from .graph_structure import GraphStructure
from .models import ASTGraphNode, ASTGraphEdge, EdgeType
from .exceptions import ExportError


class GraphExporter:
    """グラフ構造を様々な形式にエクスポートするクラス。"""
    
    def __init__(self, graph: GraphStructure):
        """
        GraphExporterの初期化。
        
        Args:
            graph: エクスポート対象のグラフ構造
        """
        self.graph = graph
    
    def export_to_dict(self, 
                      include_metadata: bool = True,
                      include_source_info: bool = True) -> Dict[str, Any]:
        """
        グラフをPython辞書形式に変換する。
        
        Args:
            include_metadata: グラフのメタデータを含めるか
            include_source_info: ソース情報を含めるか
            
        Returns:
            グラフデータを含む辞書
            
        Raises:
            ExportError: エクスポート中にエラーが発生した場合
        """
        try:
            result: Dict[str, Any] = {
                "version": "1.0.0",
                "nodes": self._export_nodes(),
                "edges": self._export_edges(),
            }
            
            if include_metadata:
                result["metadata"] = self._export_metadata()
            
            if include_source_info and self.graph.source_info:
                result["source_info"] = self._export_source_info()
            
            return result
            
        except Exception as e:
            raise ExportError(f"Failed to export graph to dict: {str(e)}") from e
    
    def export_to_json(self, 
                      indent: Optional[int] = 2,
                      include_metadata: bool = True,
                      include_source_info: bool = True) -> str:
        """
        グラフをJSON文字列に変換する。
        
        Args:
            indent: インデントレベル（Noneで圧縮形式）
            include_metadata: グラフのメタデータを含めるか
            include_source_info: ソース情報を含めるか
            
        Returns:
            JSON形式の文字列
            
        Raises:
            ExportError: エクスポート中にエラーが発生した場合
        """
        try:
            data = self.export_to_dict(include_metadata, include_source_info)
            return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True)
            
        except Exception as e:
            raise ExportError(f"Failed to export graph to JSON: {str(e)}") from e
    
    def export_to_file(self,
                      file_path: str,
                      indent: Optional[int] = 2,
                      include_metadata: bool = True,
                      include_source_info: bool = True) -> None:
        """
        グラフをJSONファイルに出力する。
        
        Args:
            file_path: 出力先ファイルパス
            indent: インデントレベル（Noneで圧縮形式）
            include_metadata: グラフのメタデータを含めるか
            include_source_info: ソース情報を含めるか
            
        Raises:
            ExportError: ファイル出力中にエラーが発生した場合
        """
        try:
            json_data = self.export_to_json(indent, include_metadata, include_source_info)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
                
        except IOError as e:
            raise ExportError(f"Failed to write to file {file_path}: {str(e)}") from e
        except Exception as e:
            raise ExportError(f"Failed to export graph to file: {str(e)}") from e
    
    def export_to_stream(self,
                        stream: IO[str],
                        indent: Optional[int] = 2,
                        include_metadata: bool = True,
                        include_source_info: bool = True) -> None:
        """
        グラフをストリームに出力する（大規模データ対応）。
        
        Args:
            stream: 出力先ストリーム
            indent: インデントレベル（Noneで圧縮形式）
            include_metadata: グラフのメタデータを含めるか
            include_source_info: ソース情報を含めるか
            
        Raises:
            ExportError: ストリーム出力中にエラーが発生した場合
        """
        try:
            # ストリーミング対応のため、部分的に出力
            stream.write('{\n')
            stream.write(f'  "version": "1.0.0",\n')
            
            # ノードをストリーミング出力
            stream.write('  "nodes": [\n')
            nodes = list(self.graph.nodes.values())
            for i, node in enumerate(nodes):
                node_dict = self._node_to_dict(node)
                json_str = json.dumps(node_dict, indent=indent, ensure_ascii=False)
                # インデント調整
                if indent:
                    json_str = '\n'.join('    ' + line for line in json_str.split('\n'))
                stream.write(json_str)
                if i < len(nodes) - 1:
                    stream.write(',')
                stream.write('\n')
            stream.write('  ],\n')
            
            # エッジをストリーミング出力
            stream.write('  "edges": [\n')
            edges = list(self.graph.edges)
            for i, edge in enumerate(edges):
                edge_dict = self._edge_to_dict(edge)
                json_str = json.dumps(edge_dict, indent=indent, ensure_ascii=False)
                # インデント調整
                if indent:
                    json_str = '\n'.join('    ' + line for line in json_str.split('\n'))
                stream.write(json_str)
                if i < len(edges) - 1:
                    stream.write(',')
                stream.write('\n')
            stream.write('  ]')
            
            # メタデータとソース情報
            if include_metadata:
                stream.write(',\n')
                metadata = self._export_metadata()
                stream.write(f'  "metadata": {json.dumps(metadata, indent=indent)}')
            
            if include_source_info and self.graph.source_info:
                stream.write(',\n')
                source_info = self._export_source_info()
                stream.write(f'  "source_info": {json.dumps(source_info, indent=indent)}')
            
            stream.write('\n}\n')
            stream.flush()
            
        except Exception as e:
            raise ExportError(f"Failed to export graph to stream: {str(e)}") from e
    
    def _export_nodes(self) -> List[Dict[str, Any]]:
        """全ノードを辞書のリストに変換する。"""
        return [self._node_to_dict(node) for node in self.graph.nodes.values()]
    
    def _export_edges(self) -> List[Dict[str, Any]]:
        """全エッジを辞書のリストに変換する。"""
        return [self._edge_to_dict(edge) for edge in self.graph.edges]
    
    def _node_to_dict(self, node: ASTGraphNode) -> Dict[str, Any]:
        """ノードを辞書に変換する。"""
        return {
            "id": node.node_id,
            "type": node.node_type,
            "label": node.label,
            "ast_node_info": node.ast_node_info,
            "source_location": node.source_location,
            "metadata": node.metadata or {}
        }
    
    def _edge_to_dict(self, edge: ASTGraphEdge) -> Dict[str, Any]:
        """エッジを辞書に変換する。"""
        return {
            "id": edge.edge_id,
            "source": edge.source_id,
            "target": edge.target_id,
            "edge_type": edge.edge_type.value,
            "label": edge.label,
            "metadata": edge.metadata or {}
        }
    
    def _export_metadata(self) -> Dict[str, Any]:
        """グラフのメタデータをエクスポート用に変換する。"""
        return {
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "created_at": datetime.now().isoformat(),
            "export_id": str(uuid.uuid4()),
            "edge_types": list(set(edge.edge_type.value for edge in self.graph.edges))
        }
    
    def _export_source_info(self) -> Dict[str, Any]:
        """ソース情報をエクスポート用に変換する。"""
        if not self.graph.source_info:
            return {}
        
        return {
            "source_id": self.graph.source_info.source_id,
            "file_path": self.graph.source_info.file_path,
            "file_hash": self.graph.source_info.file_hash,
            "parsed_at": self.graph.source_info.parsed_at.isoformat(),
            "encoding": self.graph.source_info.encoding,
            "line_count": self.graph.source_info.line_count,
            "size_bytes": self.graph.source_info.size_bytes
        }