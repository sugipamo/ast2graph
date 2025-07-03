"""Exception hierarchy for ast2graph."""

from typing import Optional, Any, Dict


class AST2GraphError(Exception):
    """Base exception class for ast2graph."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception with message and optional details."""
        super().__init__(message)
        self.details = details or {}


class ValidationError(AST2GraphError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None) -> None:
        """Initialize validation error with field and value information."""
        details = {}
        if field is not None:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, details)


class NodeNotFoundError(AST2GraphError):
    """Raised when a referenced node is not found in the graph."""
    
    def __init__(self, node_id: str, context: Optional[str] = None) -> None:
        """Initialize with the missing node ID."""
        message = f"Node not found: {node_id}"
        if context:
            message += f" (context: {context})"
        super().__init__(message, {"node_id": node_id, "context": context})


class ParseError(AST2GraphError):
    """Raised when source code parsing fails."""
    
    def __init__(
        self, 
        message: str, 
        line_no: Optional[int] = None, 
        col_offset: Optional[int] = None,
        file_path: Optional[str] = None
    ) -> None:
        """Initialize parse error with location information."""
        details = {}
        if line_no is not None:
            details["line_no"] = line_no
        if col_offset is not None:
            details["col_offset"] = col_offset
        if file_path is not None:
            details["file_path"] = file_path
        
        location_parts = []
        if file_path:
            location_parts.append(f"file: {file_path}")
        if line_no is not None:
            location_parts.append(f"line: {line_no}")
        if col_offset is not None:
            location_parts.append(f"column: {col_offset}")
        
        if location_parts:
            message = f"{message} ({', '.join(location_parts)})"
        
        super().__init__(message, details)
        self.line_no = line_no
        self.col_offset = col_offset
        self.file_path = file_path


class FileReadError(AST2GraphError):
    """Raised when file reading operations fail."""
    
    def __init__(self, file_path: str, error: Exception) -> None:
        """Initialize file read error with path and original error."""
        message = f"Failed to read file: {file_path}"
        if isinstance(error, FileNotFoundError):
            message = f"File not found: {file_path}"
        elif isinstance(error, PermissionError):
            message = f"Permission denied: {file_path}"
        elif isinstance(error, UnicodeDecodeError):
            message = f"Encoding error in file: {file_path}"
        else:
            message = f"{message} ({type(error).__name__}: {str(error)})"
        
        super().__init__(message, {"file_path": file_path, "error_type": type(error).__name__})
        self.original_error = error


class GraphBuildError(AST2GraphError):
    """Raised when graph construction fails."""
    
    def __init__(self, message: str, node_type: Optional[str] = None, ast_dump: Optional[str] = None) -> None:
        """Initialize with node type and AST dump for debugging."""
        details = {}
        if node_type is not None:
            details["node_type"] = node_type
        if ast_dump is not None:
            details["ast_dump"] = ast_dump
        super().__init__(message, details)


class NodeLimitExceeded(AST2GraphError):
    """Raised when the number of nodes exceeds the configured limit."""
    
    def __init__(self, limit: int, actual: int, file_path: Optional[str] = None) -> None:
        """Initialize with limit and actual count information."""
        message = f"Node limit exceeded: {actual} nodes (limit: {limit})"
        if file_path:
            message += f" in file: {file_path}"
        super().__init__(
            message, 
            {"limit": limit, "actual": actual, "file_path": file_path}
        )


class MemoryLimitExceeded(AST2GraphError):
    """Raised when memory usage exceeds the configured limit."""
    
    def __init__(self, limit_mb: float, actual_mb: float, operation: Optional[str] = None) -> None:
        """Initialize with memory limit information."""
        message = f"Memory limit exceeded: {actual_mb:.2f} MB (limit: {limit_mb:.2f} MB)"
        if operation:
            message += f" during operation: {operation}"
        super().__init__(
            message,
            {"limit_mb": limit_mb, "actual_mb": actual_mb, "operation": operation}
        )


class TimeoutError(AST2GraphError):
    """Raised when a processing operation times out."""
    
    def __init__(self, timeout_seconds: int, operation: Optional[str] = None, file_path: Optional[str] = None) -> None:
        """Initialize with timeout information."""
        message = f"Operation timed out after {timeout_seconds} seconds"
        details = {"timeout_seconds": timeout_seconds}
        
        if operation:
            message += f": {operation}"
            details["operation"] = operation
        if file_path:
            message += f" (file: {file_path})"
            details["file_path"] = file_path
        
        super().__init__(message, details)


class UnsupportedNodeTypeError(AST2GraphError):
    """Raised when an unsupported AST node type is encountered."""
    
    def __init__(self, node_type: str, ast_dump: Optional[str] = None) -> None:
        """Initialize with the unsupported node type."""
        message = f"Unsupported AST node type: {node_type}"
        details = {"node_type": node_type}
        if ast_dump:
            details["ast_dump"] = ast_dump
        super().__init__(message, details)


class ExportError(AST2GraphError):
    """Raised when graph export operations fail."""
    
    def __init__(self, message: str, format: Optional[str] = None, operation: Optional[str] = None) -> None:
        """Initialize export error with format and operation information."""
        details = {}
        if format is not None:
            details["format"] = format
        if operation is not None:
            details["operation"] = operation
        super().__init__(message, details)


class BatchProcessingError(AST2GraphError):
    """Raised when batch processing operations fail."""
    
    def __init__(
        self, 
        message: str, 
        batch_size: Optional[int] = None, 
        failed_count: Optional[int] = None,
        total_count: Optional[int] = None
    ) -> None:
        """Initialize batch processing error with batch information."""
        details = {}
        if batch_size is not None:
            details["batch_size"] = batch_size
        if failed_count is not None:
            details["failed_count"] = failed_count
        if total_count is not None:
            details["total_count"] = total_count
        
        if failed_count is not None and total_count is not None:
            message += f" ({failed_count}/{total_count} files failed)"
        
        super().__init__(message, details)