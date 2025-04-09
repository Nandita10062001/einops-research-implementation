#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage examples for the einops_modified module.
This file demonstrates the functionality described in the assignment.
"""

import numpy as np
import time
import psutil
from einops_modified import rearrange, EinopsError, PatternError, ShapeError, MemoryError

def main():
    """Run examples of the einops_modified functionality."""
    print("Einops Modified Usage Examples\n")
    
    # Example 1: Transpose
    print("Example 1: Transpose")
    x = np.random.rand(3, 4)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, 'h w -> w h')
    print(f"Result shape: {result.shape}")
    print(f"Result equals transpose: {np.array_equal(result, x.T)}\n")
    
    # Example 2: Split an axis
    print("Example 2: Split an axis")
    x = np.random.rand(12, 10)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '(h w) c -> h w c', h=3)
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (3, 4, 10)\n")
    
    # Example 3: Merge axes
    print("Example 3: Merge axes")
    x = np.random.rand(3, 4, 5)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, 'a b c -> (a b) c')
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (12, 5)\n")
    
    # Example 4: Repeat an axis
    print("Example 4: Repeat an axis")
    x = np.random.rand(3, 1, 5)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, 'a 1 c -> a b c', b=5)
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (3, 5, 5)\n")
    
    # Example 5: Handle batch dimensions
    print("Example 5: Handle batch dimensions")
    x = np.random.rand(2, 3, 4, 5)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '... h w -> ... (h w)')
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (2, 3, 20)\n")
    
    # Example 6: Complex operation with multiple transformations
    print("Example 6: Complex operation with multiple transformations")
    x = np.random.rand(2, 3, 4, 5)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '... h w -> ... w h')
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (2, 3, 5, 4)\n")
    
    # Example 7: Error handling - invalid pattern
    print("Example 7: Error handling - invalid pattern")
    x = np.random.rand(3, 4)
    print(f"Original shape: {x.shape}")
    try:
        result = rearrange(x, 'invalid pattern')
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 8: Error handling - shape mismatch
    print("Example 8: Error handling - shape mismatch")
    x = np.random.rand(3, 4)
    print(f"Original shape: {x.shape}")
    try:
        result = rearrange(x, '(h w) c -> h w c', h=5)
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 9: Error handling - nested parentheses
    print("Example 9: Error handling - nested parentheses")
    x = np.random.rand(3, 4, 5)
    print(f"Original shape: {x.shape}")
    try:
        result = rearrange(x, '((a b) c) -> a b c')
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 10: Error handling - unmatched parentheses
    print("Example 10: Error handling - unmatched parentheses")
    x = np.random.rand(3, 4, 5)
    print(f"Original shape: {x.shape}")
    try:
        result = rearrange(x, '(a b c -> a b c')
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 11: Error handling - memory usage
    print("Example 11: Error handling - memory usage")
    try:
        # Create a large tensor
        x = np.random.rand(1000, 1000, 1000)
        print(f"Original shape: {x.shape}")
        result = rearrange(x, 'a b c -> (a b) c')
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 12: Edge cases - empty tensor
    print("Example 12: Edge cases - empty tensor")
    try:
        x = np.array([])
        print(f"Original shape: {x.shape}")
        result = rearrange(x, 'a -> a')
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 13: Edge cases - single dimension
    print("Example 13: Edge cases - single dimension")
    x = np.array([1, 2, 3])
    print(f"Original shape: {x.shape}")
    result = rearrange(x, 'a -> a')
    print(f"Result shape: {result.shape}")
    print(f"Result equals original: {np.array_equal(result, x)}\n")
    
    # Example 14: Edge cases - zero dimensions
    print("Example 14: Edge cases - zero dimensions")
    x = np.array(42)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '->')
    print(f"Result shape: {result.shape}")
    print(f"Result equals original: {np.array_equal(result, x)}\n")
    
    # Example 15: Performance
    print("Example 15: Performance")
    # Create a moderately large tensor
    x = np.random.rand(100, 100, 100)
    print(f"Original shape: {x.shape}")
    
    # Measure time for common operations
    start = time.time()
    result = rearrange(x, 'a b c -> c b a')
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Result shape: {result.shape}\n")
    
    # Example 16: Complex patterns - multiple splits and merges
    print("Example 16: Complex patterns - multiple splits and merges")
    x = np.random.rand(24, 10, 5)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '(h w) c d -> h w (c d)', h=4)
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (4, 6, 50)\n")
    
    # Example 17: Complex patterns - nested operations with ellipsis
    print("Example 17: Complex patterns - nested operations with ellipsis")
    x = np.random.rand(2, 3, 12, 5)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '... (h w) c -> ... w h c', h=3)
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (2, 3, 4, 3, 5)\n")
    
    # Example 18: Complex patterns - multiple ellipsis
    print("Example 18: Complex patterns - multiple ellipsis")
    x = np.random.rand(2, 3, 4, 5, 6)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, '... a b ... -> ... b a ...')
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: (2, 3, 5, 4, 6)\n")
    
    # Example 19: Data types - float32
    print("Example 19: Data types - float32")
    x_float32 = np.random.rand(3, 4).astype(np.float32)
    print(f"Original dtype: {x_float32.dtype}")
    result_float32 = rearrange(x_float32, 'h w -> w h')
    print(f"Result dtype: {result_float32.dtype}\n")
    
    # Example 20: Data types - int64
    print("Example 20: Data types - int64")
    x_int64 = np.random.randint(0, 100, size=(3, 4), dtype=np.int64)
    print(f"Original dtype: {x_int64.dtype}")
    result_int64 = rearrange(x_int64, 'h w -> w h')
    print(f"Result dtype: {result_int64.dtype}\n")
    
    # Example 21: Data types - boolean
    print("Example 21: Data types - boolean")
    x_bool = np.random.choice([True, False], size=(3, 4))
    print(f"Original dtype: {x_bool.dtype}")
    result_bool = rearrange(x_bool, 'h w -> w h')
    print(f"Result dtype: {result_bool.dtype}\n")
    
    # Example 22: Edge cases - single element tensor
    print("Example 22: Edge cases - single element tensor")
    x = np.array([[[1]]])
    print(f"Original shape: {x.shape}")
    result = rearrange(x, 'a b c -> c b a')
    print(f"Result shape: {result.shape}")
    print(f"Result value: {result[0, 0, 0]}\n")
    
    # Example 23: Edge cases - zero-sized dimension
    print("Example 23: Edge cases - zero-sized dimension")
    try:
        x = np.random.rand(0, 3, 4)
        print(f"Original shape: {x.shape}")
        result = rearrange(x, 'a b c -> c b a')
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    # Example 24: Edge cases - very large dimensions
    print("Example 24: Edge cases - very large dimensions")
    x = np.random.rand(1000, 1000)
    print(f"Original shape: {x.shape}")
    result = rearrange(x, 'a b -> b a')
    print(f"Result shape: {result.shape}\n")
    
    # Example 25: Edge cases - negative dimensions
    print("Example 25: Edge cases - negative dimensions")
    try:
        x = np.random.rand(3, 4)
        print(f"Original shape: {x.shape}")
        result = rearrange(x, 'a b -> b a', a=-1)
        print("This should not be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {str(e)}\n")
    
    print("All examples completed successfully!")

if __name__ == "__main__":
    main() 