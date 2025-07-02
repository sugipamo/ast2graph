"""AST parsing functionality for Python source code."""
import ast as ast_module
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .exceptions import ParseError, FileReadError


@dataclass
class ParseResult:
    """Result of parsing operation."""
    
    ast: Optional[ast_module.AST] = None
    error: Optional[Exception] = None
    source_code: str = ""
    file_path: Optional[str] = None
    encoding: str = "utf-8"
    line_count: int = 0
    source_hash: str = ""
    file_hash: Optional[str] = None
    parsed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ASTParser:
    """Parser for Python source code to AST."""
    
    def __init__(self):
        """Initialize the AST parser."""
        self._encoding_pattern = re.compile(
            rb"coding[=:]\s*([-\w.]+)", re.IGNORECASE
        )
    
    def parse_code(self, source_code: str, filename: str = "<string>") -> ParseResult:
        """
        Parse Python source code string into AST.
        
        Args:
            source_code: Python source code as string
            filename: Optional filename for error reporting
            
        Returns:
            ParseResult containing AST or error information
        """
        result = ParseResult(
            source_code=source_code,
            line_count=source_code.count('\n') + (1 if source_code else 0),
            source_hash=self._compute_hash(source_code.encode()),
            encoding='utf-8'  # String input is always UTF-8 in Python 3
        )
        
        try:
            # Parse the source code
            tree = ast_module.parse(source_code, filename=filename, mode='exec')
            result.ast = tree
            
        except SyntaxError as e:
            # Extract syntax error details
            error = ParseError(
                message=str(e.msg) if hasattr(e, 'msg') else str(e),
                line_no=e.lineno,
                col_offset=e.offset,
                file_path=filename if filename != "<string>" else None
            )
            result.error = error
            
        except Exception as e:
            # Handle any other parsing errors
            result.error = ParseError(f"Failed to parse source code: {str(e)}")
        
        return result
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        Parse Python source file into AST.
        
        Args:
            file_path: Path to Python source file
            
        Returns:
            ParseResult containing AST or error information
        """
        path = Path(file_path)
        result = ParseResult(file_path=file_path)
        
        try:
            # Read file content
            source_bytes = path.read_bytes()
            result.file_hash = self._compute_hash(source_bytes)
            
            # Detect encoding
            encoding = self._detect_encoding(source_bytes)
            result.encoding = encoding
            
            # Decode source code
            source_code = source_bytes.decode(encoding)
            result.source_code = source_code
            result.line_count = source_code.count('\n') + (1 if source_code else 0)
            result.source_hash = self._compute_hash(source_code.encode())
            
            # Parse the code
            tree = ast_module.parse(source_code, filename=file_path, mode='exec')
            result.ast = tree
            
        except (FileNotFoundError, PermissionError, OSError) as e:
            result.error = FileReadError(file_path, e)
            
        except UnicodeDecodeError as e:
            result.error = FileReadError(file_path, e)
            
        except SyntaxError as e:
            # Extract syntax error details
            error = ParseError(
                message=str(e.msg) if hasattr(e, 'msg') else str(e),
                line_no=e.lineno,
                col_offset=e.offset,
                file_path=file_path
            )
            result.error = error
            
        except Exception as e:
            # Handle any other errors
            result.error = ParseError(
                f"Failed to parse file: {str(e)}",
                file_path=file_path
            )
        
        return result
    
    def _detect_encoding(self, source_bytes: bytes) -> str:
        """
        Detect encoding of Python source file.
        
        Checks for:
        1. UTF-8 BOM
        2. Encoding declaration in first two lines
        3. Default to UTF-8
        
        Args:
            source_bytes: Raw bytes from source file
            
        Returns:
            Detected encoding name
        """
        # Check for UTF-8 BOM
        if source_bytes.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        
        # Check first two lines for encoding declaration
        lines = source_bytes.split(b'\n', 2)[:2]
        for line in lines:
            match = self._encoding_pattern.search(line)
            if match:
                encoding = match.group(1).decode('ascii')
                # Normalize encoding name
                return encoding.lower()
        
        # Default to UTF-8
        return 'utf-8'
    
    def _compute_hash(self, data: bytes) -> str:
        """
        Compute SHA-256 hash of data.
        
        Args:
            data: Bytes to hash
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(data).hexdigest()