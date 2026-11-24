#!/usr/bin/env python3
"""Inspect columnar parquet files in detail"""

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import sys
import os

def inspect_parquet_detailed(file_path):
    """Inspect a parquet file and show detailed contents"""
    print(f"\n{'='*80}")
    print(f"📊 Inspecting: {file_path}")
    print(f"{'='*80}")
    
    try:
        # Read the parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        print(f"\n📋 Schema:")
        for field in table.schema:
            print(f"  - {field.name}: {field.type}")
        
        print(f"\n📊 Shape: {df.shape} (rows, columns)")
        print(f"📊 Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"\n📊 First few rows:")
            print(df.head())
            
            # Inspect vector column specifically
            if 'vector' in df.columns:
                print(f"\n🔍 Vector column details:")
                first_vector = df['vector'].iloc[0]
                print(f"  - Type of first vector: {type(first_vector)}")
                print(f"  - First vector shape: {np.array(first_vector).shape if first_vector is not None else 'None'}")
                print(f"  - First 10 values: {first_vector[:10] if first_vector is not None else 'None'}")
                
                # Check if all vectors have same dimension
                dimensions = [len(v) if v is not None else 0 for v in df['vector']]
                unique_dims = set(dimensions)
                print(f"  - Unique dimensions: {unique_dims}")
            
            # Show data types
            print(f"\n📊 Data types:")
            print(df.dtypes)
            
            # Show null counts
            print(f"\n📊 Null counts:")
            print(df.isnull().sum())
            
        else:
            print("\n⚠️ DataFrame is empty!")
            
        # Also read raw Arrow data to see exact structure
        print(f"\n📊 Raw Arrow RecordBatch inspection:")
        reader = pq.ParquetFile(file_path)
        for i, batch in enumerate(reader.iter_batches()):
            print(f"\n  Batch {i}:")
            print(f"    - Num rows: {batch.num_rows}")
            print(f"    - Num columns: {batch.num_columns}")
            print(f"    - Column names: {batch.column_names}")
            
            # Check vector column structure
            if 'vector' in batch.column_names:
                vector_col = batch.column('vector')
                print(f"    - Vector column type: {vector_col.type}")
                print(f"    - Vector column length: {len(vector_col)}")
                if len(vector_col) > 0:
                    # Try to access first element
                    try:
                        first_elem = vector_col[0]
                        print(f"    - First vector element type: {type(first_elem)}")
                        print(f"    - First vector values: {first_elem.as_py()[:5] if hasattr(first_elem, 'as_py') else str(first_elem)[:50]}")
                    except Exception as e:
                        print(f"    - Error accessing first element: {e}")
                        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Find recent test files
    import subprocess
    import re
    
    # Run the test to get file path
    print("🚀 Running the debug test to generate files...")
    result = subprocess.run(
        ["cargo", "test", "viper::tests::debug_compaction_test::test_viper_flush_and_compaction_debug", "--", "--nocapture"],
        capture_output=True,
        text=True
    )
    
    # Extract file paths from output
    flushed_files = []
    compacted_files = []
    
    for line in result.stdout.split('\n'):
        if "Reading file://" in line:
            match = re.search(r'file://(/[^\s]+\.parquet)', line)
            if match:
                file_path = match.group(1)
                if 'partition_' in file_path:
                    flushed_files.append(file_path)
                elif 'compacted_' in file_path:
                    compacted_files.append(file_path)
    
    print(f"\n🔍 Found {len(flushed_files)} flushed files and {len(compacted_files)} compacted files")
    
    # Inspect flushed files
    for file_path in flushed_files:
        if os.path.exists(file_path):
            print(f"\n{'*'*80}")
            print("FLUSHED FILE:")
            inspect_parquet_detailed(file_path)
    
    # Inspect compacted files
    for file_path in compacted_files:
        if os.path.exists(file_path):
            print(f"\n{'*'*80}")
            print("COMPACTED FILE:")
            inspect_parquet_detailed(file_path)