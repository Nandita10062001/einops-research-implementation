import numpy as np
import pytest
from einops_modified import rearrange, EinopsError, PatternError, ShapeError, MemoryError

def test_basic_transpose():
    """Test basic transposition operation."""
    x = np.random.rand(3, 4)
    result = rearrange(x, 'h w -> w h')
    assert result.shape == (4, 3)
    assert np.array_equal(result, x.T)

def test_split_axis():
    """Test splitting an axis into multiple dimensions."""
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    assert result.shape == (3, 4, 10)

def test_merge_axes():
    """Test merging multiple axes into one."""
    x = np.random.rand(3, 4, 5)
    result = rearrange(x, 'a b c -> (a b) c')
    assert result.shape == (12, 5)

def test_repeat_axis():
    """Test repeating an axis."""
    x = np.random.rand(3, 1, 5)
    result = rearrange(x, 'a 1 c -> a b c', b=4)
    assert result.shape == (3, 4, 5)

def test_invalid_pattern():
    """Test handling of invalid pattern strings."""
    x = np.random.rand(3, 4)
    with pytest.raises(PatternError) as exc_info:
        rearrange(x, 'invalid pattern')
    assert "Invalid pattern format" in str(exc_info.value)

def test_shape_mismatch():
    """Test handling of shape mismatches."""
    x = np.random.rand(3, 4)
    with pytest.raises(ShapeError) as exc_info:
        rearrange(x, '(h w) c -> h w c', h=5)
    assert "Cannot split dimension" in str(exc_info.value)

def test_complex_operations():
    """Test complex operations with multiple transformations."""
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, '... h w -> ... (h w)')
    assert result.shape == (2, 3, 20)

def test_ellipsis():
    """Test handling of ellipsis for batch dimensions."""
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, '... h w -> ... w h')
    assert result.shape == (2, 3, 5, 4)

def test_nested_parentheses():
    """Test handling of nested parentheses (should fail)."""
    x = np.random.rand(3, 4, 5)
    with pytest.raises(PatternError) as exc_info:
        rearrange(x, '((a b) c) -> a b c')
    assert "Nested parentheses not allowed" in str(exc_info.value)

def test_unmatched_parentheses():
    """Test handling of unmatched parentheses."""
    x = np.random.rand(3, 4, 5)
    with pytest.raises(PatternError) as exc_info:
        rearrange(x, '(a b c -> a b c')
    assert "Unmatched" in str(exc_info.value)

def test_memory_usage():
    """Test memory usage estimation."""
    # Create a large tensor
    x = np.random.rand(1000, 1000, 1000)
    with pytest.raises(MemoryError) as exc_info:
        rearrange(x, 'a b c -> (a b) c')
    assert "Operation would require" in str(exc_info.value)

def test_edge_cases():
    """Test various edge cases."""
    # Empty tensor
    x = np.array([])
    with pytest.raises(ShapeError):
        rearrange(x, 'a -> a')
    
    # Single dimension
    x = np.array([1, 2, 3])
    result = rearrange(x, 'a -> a')
    assert np.array_equal(result, x)
    
    # Zero dimensions
    x = np.array(42)
    result = rearrange(x, '->')
    assert np.array_equal(result, x)

def test_performance():
    """Test performance with large tensors."""
    # Create a moderately large tensor
    x = np.random.rand(100, 100, 100)
    
    # Measure time for common operations
    import time
    start = time.time()
    result = rearrange(x, 'a b c -> c b a')
    end = time.time()
    assert end - start < 1.0  # Should complete within 1 second
    
    # Test memory efficiency
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss
    result = rearrange(x, 'a b c -> (a b) c')
    mem_after = process.memory_info().rss
    assert mem_after - mem_before < 1024 * 1024 * 100  # Less than 100MB increase

def test_complex_patterns():
    """Test more complex pattern combinations."""
    # Test multiple splits and merges
    x = np.random.rand(24, 10, 5)
    result = rearrange(x, '(h w) c d -> h w (c d)', h=4)
    assert result.shape == (4, 6, 50)
    
    # Test nested operations with ellipsis
    x = np.random.rand(2, 3, 12, 5)
    result = rearrange(x, '... (h w) c -> ... w h c', h=3)
    assert result.shape == (2, 3, 4, 3, 5)
    
    # Test multiple ellipsis
    x = np.random.rand(2, 3, 4, 5, 6)
    result = rearrange(x, '... a b ... -> ... b a ...')
    assert result.shape == (2, 3, 5, 4, 6)

def test_data_types():
    """Test different numpy data types."""
    # Test with float32
    x = np.random.rand(3, 4).astype(np.float32)
    result = rearrange(x, 'h w -> w h')
    assert result.dtype == np.float32
    
    # Test with int64
    x = np.random.randint(0, 100, size=(3, 4), dtype=np.int64)
    result = rearrange(x, 'h w -> w h')
    assert result.dtype == np.int64
    
    # Test with bool
    x = np.random.choice([True, False], size=(3, 4))
    result = rearrange(x, 'h w -> w h')
    assert result.dtype == bool

def test_edge_cases_extended():
    """Test additional edge cases."""
    # Test with single element tensor
    x = np.array([[[1]]])
    result = rearrange(x, 'a b c -> c b a')
    assert result.shape == (1, 1, 1)
    assert result[0, 0, 0] == 1
    
    # Test with zero-sized dimension
    x = np.random.rand(0, 3, 4)
    with pytest.raises(ShapeError) as exc_info:
        rearrange(x, 'a b c -> c b a')
    assert "Cannot rearrange empty tensors" in str(exc_info.value)
    
    # Test with very large dimensions
    x = np.random.rand(1000, 1000)
    result = rearrange(x, 'a b -> b a')
    assert result.shape == (1000, 1000)
    
    # Test with negative dimensions (should fail)
    x = np.random.rand(3, 4)
    with pytest.raises(ShapeError) as exc_info:
        rearrange(x, 'a b -> b a', a=-1)
    assert "Invalid dimension size" in str(exc_info.value)

if __name__ == '__main__':
    pytest.main([__file__])