"""Data types for batch processing."""
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from datetime import datetime

from .graph_structure import GraphStructure


@dataclass
class FailedFile:
    """Information about a file that failed to process."""
    
    file_path: str
    error: Exception
    error_type: str
    error_message: str
    
    @classmethod
    def from_exception(cls, file_path: str, error: Exception) -> 'FailedFile':
        """Create FailedFile from an exception."""
        return cls(
            file_path=file_path,
            error=error,
            error_type=type(error).__name__,
            error_message=str(error)
        )


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    
    total_files: int
    processed_files: int
    failed_files: List[FailedFile]
    processing_time: float
    memory_peak_mb: float
    graphs: List[GraphStructure]
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of processing."""
        if self.total_files == 0:
            return 1.0
        return self.processed_files / self.total_files
    
    @staticmethod
    def merge(results: List['BatchResult']) -> 'BatchResult':
        """Merge multiple batch results into one."""
        if not results:
            return BatchResult(0, 0, [], 0.0, 0.0, [])
        
        total_files = sum(r.total_files for r in results)
        processed_files = sum(r.processed_files for r in results)
        failed_files = []
        for r in results:
            failed_files.extend(r.failed_files)
        processing_time = sum(r.processing_time for r in results)
        memory_peak_mb = max(r.memory_peak_mb for r in results) if results else 0.0
        graphs = []
        for r in results:
            graphs.extend(r.graphs)
        
        return BatchResult(
            total_files=total_files,
            processed_files=processed_files,
            failed_files=failed_files,
            processing_time=processing_time,
            memory_peak_mb=memory_peak_mb,
            graphs=graphs
        )


@dataclass
class ProcessingProgress:
    """Current progress of batch processing."""
    
    current_file: str
    files_completed: int
    files_total: int
    current_phase: str  # "parsing", "building", "exporting"
    elapsed_time: float
    estimated_remaining: float
    memory_usage_mb: float
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.files_total == 0:
            return 100.0
        return (self.files_completed / self.files_total) * 100.0


@dataclass
class ProcessingMetrics:
    """Detailed metrics from processing operation."""
    
    files_processed: int
    nodes_created: int
    edges_created: int
    processing_time: float
    memory_peak: float
    errors: List[str]
    start_time: datetime
    end_time: datetime
    
    @property
    def average_nodes_per_file(self) -> float:
        """Calculate average nodes created per file."""
        if self.files_processed == 0:
            return 0.0
        return self.nodes_created / self.files_processed
    
    @property
    def files_per_second(self) -> float:
        """Calculate processing speed in files per second."""
        if self.processing_time == 0:
            return 0.0
        return self.files_processed / self.processing_time


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_size: int = 50
    max_workers: Optional[int] = None
    memory_limit_mb: int = 500
    use_multiprocessing: bool = False
    progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    error_callback: Optional[Callable[[FailedFile], None]] = None
    enable_profiling: bool = False
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        
        if self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")