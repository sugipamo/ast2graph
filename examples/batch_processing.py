#!/usr/bin/env python3
"""
Batch processing examples for ast2graph library.
"""

import os
import time
from pathlib import Path
from ast2graph import parse_directory, parse_files_stream
from ast2graph.batch_processor import BatchProcessor
from ast2graph.batch_types import ProcessingOptions, ErrorRecoveryStrategy


def example_parse_directory():
    """Example: Parse an entire directory."""
    print("=== Parse Directory Example ===")
    
    # Parse the ast2graph source directory
    src_dir = Path(__file__).parent.parent / "src" / "ast2graph"
    
    if not src_dir.exists():
        print(f"Directory not found: {src_dir}")
        return
    
    start_time = time.time()
    
    # Parse all Python files in the directory
    graphs = parse_directory(
        str(src_dir),
        include_dependencies=True,
        file_pattern="*.py"  # Only .py files in direct directory
    )
    
    elapsed = time.time() - start_time
    
    print(f"Directory: {src_dir}")
    print(f"Files processed: {len(graphs)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    
    # Show statistics for each file
    total_nodes = 0
    total_edges = 0
    
    print("\nFile statistics:")
    for file_path, graph in sorted(graphs.items()):
        file_name = os.path.basename(file_path)
        stats = graph.statistics()
        total_nodes += stats['total_nodes']
        total_edges += stats['total_edges']
        print(f"  {file_name}: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    
    print(f"\nTotal: {total_nodes} nodes, {total_edges} edges")


def example_streaming_processing():
    """Example: Stream processing for memory efficiency."""
    print("\n=== Streaming Processing Example ===")
    
    # Get list of Python files
    src_dir = Path(__file__).parent.parent / "src" / "ast2graph"
    py_files = list(src_dir.glob("*.py"))
    
    if not py_files:
        print("No Python files found")
        return
    
    print(f"Processing {len(py_files)} files with streaming...")
    
    total_nodes = 0
    total_edges = 0
    file_count = 0
    
    # Process files one at a time
    for file_path, graph in parse_files_stream([str(f) for f in py_files]):
        file_count += 1
        stats = graph.statistics()
        total_nodes += stats['total_nodes']
        total_edges += stats['total_edges']
        
        print(f"  [{file_count}/{len(py_files)}] Processed {os.path.basename(file_path)}")
        # Graph is released from memory after this iteration
    
    print(f"\nTotal processed: {total_nodes} nodes, {total_edges} edges")


def example_batch_processor():
    """Example: Advanced batch processing with options."""
    print("\n=== Batch Processor Example ===")
    
    # Get test files
    test_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_projects"
    if not test_dir.exists():
        # Fallback to examples directory
        test_dir = Path(__file__).parent
    
    py_files = list(test_dir.rglob("*.py"))[:5]  # Limit to 5 files for demo
    
    if not py_files:
        print("No test files found")
        return
    
    # Configure processing options
    options = ProcessingOptions(
        include_dependencies=True,
        max_workers=2,  # Use 2 parallel workers
        memory_limit_mb=100,  # Limit memory usage
        error_recovery=ErrorRecoveryStrategy.SKIP_AND_CONTINUE,
        enable_progress_callback=True
    )
    
    # Create batch processor
    processor = BatchProcessor(options)
    
    # Define progress callback
    def on_progress(progress):
        percent = (progress.completed / progress.total) * 100
        print(f"  Progress: {progress.completed}/{progress.total} ({percent:.1f}%)")
        if progress.current_file:
            print(f"  Current: {os.path.basename(progress.current_file)}")
    
    # Process files
    print(f"Processing {len(py_files)} files with batch processor...")
    start_time = time.time()
    
    result = processor.process_files(
        [str(f) for f in py_files],
        progress_callback=on_progress
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nBatch processing complete in {elapsed:.2f} seconds")
    print(f"Successfully processed: {len(result.graphs)}")
    print(f"Errors encountered: {len(result.errors)}")
    
    if result.errors:
        print("\nErrors:")
        for file_path, error in result.errors.items():
            print(f"  {os.path.basename(file_path)}: {error}")
    
    # Show aggregate statistics
    total_stats = result.aggregate_statistics()
    print(f"\nAggregate statistics:")
    print(f"  Total nodes: {total_stats.get('total_nodes', 0)}")
    print(f"  Total edges: {total_stats.get('total_edges', 0)}")
    print(f"  Average nodes per file: {total_stats.get('avg_nodes_per_file', 0):.1f}")


def example_error_handling():
    """Example: Handle errors during batch processing."""
    print("\n=== Error Handling Example ===")
    
    # Create a list with some invalid files
    files = [
        __file__,  # Valid file
        "non_existent_file.py",  # Does not exist
        str(Path(__file__).parent),  # Directory, not a file
    ]
    
    print(f"Processing {len(files)} files (including invalid ones)...")
    
    # Process with error recovery
    options = ProcessingOptions(
        error_recovery=ErrorRecoveryStrategy.SKIP_AND_CONTINUE
    )
    
    processor = BatchProcessor(options)
    result = processor.process_files(files)
    
    print(f"\nResults:")
    print(f"  Success: {len(result.graphs)}")
    print(f"  Errors: {len(result.errors)}")
    
    # Show successful files
    if result.graphs:
        print("\nSuccessfully processed:")
        for file_path in result.graphs:
            print(f"  ✓ {os.path.basename(file_path)}")
    
    # Show errors
    if result.errors:
        print("\nErrors encountered:")
        for file_path, error in result.errors.items():
            print(f"  ✗ {file_path}: {error}")


def example_memory_monitoring():
    """Example: Monitor memory usage during processing."""
    print("\n=== Memory Monitoring Example ===")
    
    try:
        import psutil
    except ImportError:
        print("psutil not installed. Install with: pip install psutil")
        return
    
    # Get current process
    process = psutil.Process()
    
    # Get initial memory usage
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Process some files
    src_dir = Path(__file__).parent.parent / "src" / "ast2graph"
    py_files = list(src_dir.glob("*.py"))[:3]
    
    if py_files:
        # Standard processing (all in memory)
        print("\nStandard processing:")
        graphs = parse_directory(str(src_dir), file_pattern="*.py")
        
        after_standard = process.memory_info().rss / 1024 / 1024
        print(f"Memory after standard: {after_standard:.1f} MB")
        print(f"Memory increase: {after_standard - initial_memory:.1f} MB")
        
        # Clear graphs
        del graphs
        
        # Streaming processing
        print("\nStreaming processing:")
        for file_path, graph in parse_files_stream([str(f) for f in py_files]):
            pass  # Process and immediately discard
        
        after_streaming = process.memory_info().rss / 1024 / 1024
        print(f"Memory after streaming: {after_streaming:.1f} MB")
        print(f"Memory increase: {after_streaming - initial_memory:.1f} MB")


if __name__ == "__main__":
    # Run all examples
    example_parse_directory()
    example_streaming_processing()
    example_batch_processor()
    example_error_handling()
    example_memory_monitoring()