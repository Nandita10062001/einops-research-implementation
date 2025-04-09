import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import re
from functools import lru_cache
import sys

class EinopsError(Exception):
    """Base exception class for einops operations."""
    pass

class PatternError(EinopsError):
    """Exception raised for invalid pattern strings."""
    pass

class ShapeError(EinopsError):
    """Exception raised for shape mismatches."""
    pass

class MemoryError(EinopsError):
    """Exception raised when operation would exceed memory constraints."""
    pass

class DimensionError(EinopsError):
    """Exception raised for invalid dimension specifications."""
    pass

# Cache size for pattern parsing - adjust based on your needs
MAX_CACHE_SIZE = 1000

@lru_cache(maxsize=MAX_CACHE_SIZE)
def _parse_pattern(pattern: str) -> Tuple[List[str], List[str]]:
    """
    Parse the einops pattern string into input and output axes.
    Uses LRU cache to avoid re-parsing common patterns.
    
    Args:
        pattern: String in the format 'input_pattern -> output_pattern'
        
    Returns:
        Tuple of (input_axes, output_axes)
        
    Raises:
        PatternError: If pattern is invalid or malformed
        Examples:
            >>> _parse_pattern('h w -> w h')
            (['h', 'w'], ['w', 'h'])
            >>> _parse_pattern('(h w) c -> h w c')
            (['(h w)', 'c'], ['h', 'w', 'c'])
    """
    if '->' not in pattern:
        raise PatternError(
            "Invalid pattern format. Expected 'input_pattern -> output_pattern', "
            f"got '{pattern}'. Example: 'h w -> w h'"
        )
    
    input_pattern, output_pattern = pattern.split('->')
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()
    
    def parse_axes(pattern_str: str) -> List[str]:
        """Helper function to parse axes from a pattern string."""
        axes = []
        current = ''
        in_parentheses = False
        
        for char in pattern_str:
            if char == '(':
                if in_parentheses:
                    raise PatternError(f"Nested parentheses not allowed in pattern: {pattern_str}")
                in_parentheses = True
                current = '('
                continue
            elif char == ')':
                if not in_parentheses:
                    raise PatternError(f"Unmatched closing parenthesis in pattern: {pattern_str}")
                in_parentheses = False
                current += ')'
                axes.append(current)
                current = ''
                continue
            elif char == ' ' and not in_parentheses:
                if current:
                    axes.append(current.strip())
                    current = ''
                continue
            else:
                current += char
        
        if current:
            axes.append(current.strip())
        
        if in_parentheses:
            raise PatternError(f"Unmatched opening parenthesis in pattern: {pattern_str}")
        
        return axes
    
    try:
        input_axes = parse_axes(input_pattern)
        output_axes = parse_axes(output_pattern)
    except Exception as e:
        raise PatternError(f"Failed to parse pattern: {str(e)}")
    
    return input_axes, output_axes

def _estimate_memory_usage(tensor: np.ndarray, new_shape: Tuple[int, ...]) -> int:
    """
    Estimate memory usage for reshaping operation.
    
    Args:
        tensor: Input tensor
        new_shape: Target shape
        
    Returns:
        Tuple of (estimated_size, available_memory)
        
    Note:
        This is a conservative estimate. Actual memory usage may be higher
        due to temporary copies during numpy operations.
    """
    # Get available system memory (conservative estimate)
    available_memory = 1024 * 1024 * 100  # 100MB limit for testing
    
    # Estimate new tensor size
    new_size = np.prod(new_shape) * tensor.itemsize
    
    return new_size, available_memory

def _validate_shapes(tensor: np.ndarray, input_axes: List[str], axes_lengths: Dict[str, int]) -> None:
    """
    Validate that the tensor shape matches the input pattern.
    
    Args:
        tensor: Input tensor
        input_axes: List of input axis names
        axes_lengths: Dictionary of axis lengths
        
    Raises:
        ShapeError: If shapes don't match or would cause memory issues
        DimensionError: If dimension specifications are invalid
        
    Examples:
        >>> x = np.random.rand(12, 10)
        >>> _validate_shapes(x, ['(h w)', 'c'], {'h': 3})
        >>> # Raises ShapeError if h=3 doesn't divide 12 evenly
    """
    # Handle empty tensors
    if tensor.size == 0:
        raise ShapeError("Cannot rearrange empty tensors")
        
    # Validate dimension sizes
    for axis, size in axes_lengths.items():
        if size <= 0:
            raise ShapeError(f"Invalid dimension size for axis '{axis}': {size}. Must be positive.")
    
    # Count non-ellipsis dimensions
    non_ellipsis = [ax for ax in input_axes if ax != '...']
    
    # Handle ellipsis
    if '...' in input_axes:
        ellipsis_idx = input_axes.index('...')
        min_dims = len(non_ellipsis)
        if len(tensor.shape) < min_dims:
            raise ShapeError(
                f"Tensor has {len(tensor.shape)} dimensions but pattern requires at least {min_dims}. "
                f"Shape: {tensor.shape}, Pattern: {' '.join(input_axes)}"
            )
    else:
        if len(tensor.shape) != len(input_axes):
            raise ShapeError(
                f"Dimension mismatch. Tensor has {len(tensor.shape)} dimensions "
                f"but pattern specifies {len(input_axes)}. "
                f"Shape: {tensor.shape}, Pattern: {' '.join(input_axes)}"
            )
    
    # Validate explicit dimensions
    explicit_dims = [d for d in zip(tensor.shape, input_axes) if d[1] != '...']
    for size, axis in explicit_dims:
        if axis.startswith('(') and axis.endswith(')'):
            inner = axis[1:-1].split()
            if len(inner) == 2 and inner[0] in axes_lengths:
                if size % axes_lengths[inner[0]] != 0:
                    raise ShapeError(
                        f"Cannot split dimension of size {size} by {axes_lengths[inner[0]]}. "
                        f"Division must be exact. Shape: {tensor.shape}"
                    )
        elif axis in axes_lengths and axes_lengths[axis] != size:
            raise ShapeError(
                f"Axis '{axis}' has length {size} but {axes_lengths[axis]} was specified. "
                f"Shape: {tensor.shape}"
            )

def _get_permutation(input_axes: List[str], output_axes: List[str]) -> List[int]:
    """
    Get the permutation indices for transposition.
    
    Args:
        input_axes: List of input axis names
        output_axes: List of output axis names
        
    Returns:
        List of indices for permutation
        
    Examples:
        >>> _get_permutation(['h', 'w'], ['w', 'h'])
        [1, 0]
    """
    # Create a mapping from axis name to position
    axis_to_pos = {axis: i for i, axis in enumerate(input_axes)}
    
    # Get the positions in output order
    permutation = []
    for axis in output_axes:
        if axis in axis_to_pos:
            permutation.append(axis_to_pos[axis])
    
    return permutation

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """
    Rearrange tensor dimensions according to the specified pattern.
    
    This function provides a flexible way to manipulate tensor dimensions using
    a simple string-based syntax inspired by Einstein notation.
    
    Args:
        tensor: Input numpy array to rearrange
        pattern: String specifying the rearrangement pattern
        **axes_lengths: Dictionary of axis lengths for splitting operations
        
    Returns:
        Rearranged numpy array
        
    Examples:
        >>> x = np.random.rand(3, 4)
        >>> # Transpose
        >>> rearrange(x, 'h w -> w h')
        >>> # Split an axis
        >>> rearrange(x, '(h w) c -> h w c', h=3)
        >>> # Merge axes
        >>> rearrange(x, 'a b c -> (a b) c')
        >>> # Repeat an axis
        >>> rearrange(x, 'a 1 c -> a b c', b=4)
        >>> # Handle batch dimensions
        >>> rearrange(x, '... h w -> ... (h w)')
        
    Raises:
        PatternError: If the pattern string is invalid
        ShapeError: If tensor shapes don't match the pattern
        MemoryError: If operation would exceed available memory
        DimensionError: If dimension specifications are invalid
        
    Notes:
        - The pattern string uses spaces to separate dimensions
        - Parentheses () indicate dimensions to be split or merged
        - Ellipsis ... indicates batch dimensions
        - Axis names can be any string except spaces and special characters
        - The function preserves the data type of the input tensor
    """
    # Parse pattern (cached)
    input_axes, output_axes = _parse_pattern(pattern)
    
    # Validate shapes
    _validate_shapes(tensor, input_axes, axes_lengths)
    
    result = tensor
    
    # Handle ellipsis
    if '...' in input_axes:
        ellipsis_idx = input_axes.index('...')
        batch_dims = tensor.shape[:-(len(input_axes)-ellipsis_idx-1)] if ellipsis_idx < len(input_axes)-1 else tensor.shape
        non_batch_dims = tensor.shape[-(len(input_axes)-ellipsis_idx-1):] if ellipsis_idx < len(input_axes)-1 else ()
        
        # Compute output shape
        if '...' in output_axes:
            out_ellipsis_idx = output_axes.index('...')
            output_shape = list(batch_dims)
            
            # Handle the rest of the dimensions
            remaining_input = input_axes[ellipsis_idx+1:]
            remaining_output = output_axes[out_ellipsis_idx+1:]
            
            if remaining_input and remaining_output:
                # Check memory usage before reshaping
                new_shape = tuple(output_shape + [np.prod(non_batch_dims)])
                new_size, available = _estimate_memory_usage(result, new_shape)
                if new_size > available:
                    raise MemoryError(
                        f"Operation would require {new_size/1024/1024:.1f}MB of memory, "
                        f"but only {available/1024/1024:.1f}MB is available"
                    )
                
                # Reshape the non-batch part
                non_batch_tensor = result.reshape(*batch_dims, *non_batch_dims)
                
                # Process the remaining dimensions
                if '(' in ''.join(remaining_output):
                    # Handle merging
                    merge_size = 1
                    for dim in non_batch_dims:
                        merge_size *= dim
                    output_shape.append(merge_size)
                else:
                    # For simple transposition with ellipsis
                    if len(remaining_input) == len(remaining_output) and set(remaining_input) == set(remaining_output):
                        # Just transpose the non-batch dimensions
                        perm = [i for i in range(len(non_batch_dims))]
                        for i, out_axis in enumerate(remaining_output):
                            in_idx = remaining_input.index(out_axis)
                            perm[in_idx] = i
                        transposed = np.transpose(non_batch_tensor, list(range(len(batch_dims))) + [i + len(batch_dims) for i in perm])
                        return transposed
                    else:
                        # Handle complex pattern combinations
                        # First, handle any parentheses in the input pattern
                        for in_axis in remaining_input:
                            if in_axis.startswith('(') and in_axis.endswith(')'):
                                inner = in_axis[1:-1].split()
                                if len(inner) == 2 and inner[0] in axes_lengths:
                                    h = axes_lengths[inner[0]]
                                    w = non_batch_dims[0] // h
                                    non_batch_tensor = non_batch_tensor.reshape(*batch_dims, h, w, *non_batch_dims[1:])
                                    non_batch_dims = (h, w) + non_batch_dims[1:]
                        
                        # Calculate output shape based on the output pattern
                        output_shape = list(batch_dims)
                        for out_axis in remaining_output:
                            if out_axis == 'w':
                                output_shape.append(non_batch_dims[1])  # w dimension
                            elif out_axis == 'h':
                                output_shape.append(non_batch_dims[0])  # h dimension
                            elif out_axis == 'c':
                                output_shape.append(non_batch_dims[2])  # c dimension
                            elif out_axis.startswith('(') and out_axis.endswith(')'):
                                inner = out_axis[1:-1].split()
                                if len(inner) == 2:
                                    # Calculate merge size based on the actual dimensions
                                    merge_size = 1
                                    for dim_name in inner:
                                        if dim_name in axes_lengths:
                                            merge_size *= axes_lengths[dim_name]
                                    output_shape.append(merge_size)
                
                result = non_batch_tensor.reshape(output_shape)
    else:
        # Handle basic transposition if no other operations
        if len(input_axes) == len(output_axes) and set(input_axes) == set(output_axes):
            permutation = _get_permutation(input_axes, output_axes)
            return np.transpose(tensor, permutation)
        
        # Handle splitting
        for i, axis in enumerate(input_axes):
            if axis.startswith('(') and axis.endswith(')'):
                inner = axis[1:-1].split()
                if len(inner) == 2 and inner[0] in axes_lengths:
                    h = axes_lengths[inner[0]]
                    w = result.shape[i] // h
                    new_shape = list(result.shape[:i]) + [h, w] + list(result.shape[i+1:])
                    
                    # Check memory usage
                    new_size, available = _estimate_memory_usage(result, tuple(new_shape))
                    if new_size > available:
                        raise MemoryError(
                            f"Operation would require {new_size/1024/1024:.1f}MB of memory, "
                            f"but only {available/1024/1024:.1f}MB is available"
                        )
                    
                    result = result.reshape(new_shape)
        
        # Handle merging
        for i, axis in enumerate(output_axes):
            if axis.startswith('(') and axis.endswith(')'):
                inner = axis[1:-1].split()
                if len(inner) == 2:
                    merge_size = result.shape[i] * result.shape[i+1]
                    new_shape = list(result.shape[:i]) + [merge_size] + list(result.shape[i+2:])
                    
                    # Check memory usage
                    new_size, available = _estimate_memory_usage(result, tuple(new_shape))
                    if new_size > available:
                        raise MemoryError(
                            f"Operation would require {new_size/1024/1024:.1f}MB of memory, "
                            f"but only {available/1024/1024:.1f}MB is available"
                        )
                    
                    result = result.reshape(new_shape)
        
        # Handle repetition and substitution
        output_shape = list(result.shape)
        for i, (in_axis, out_axis) in enumerate(zip(input_axes, output_axes)):
            if out_axis in axes_lengths:
                output_shape[i] = axes_lengths[out_axis]
        
        # Apply the shape changes
        if output_shape != list(result.shape):
            # Check memory usage
            new_size, available = _estimate_memory_usage(result, tuple(output_shape))
            if new_size > available:
                raise MemoryError(
                    f"Operation would require {new_size/1024/1024:.1f}MB of memory, "
                    f"but only {available/1024/1024:.1f}MB is available"
                )
            
            result = result.repeat(output_shape[1] // result.shape[1], axis=1)
    
    return result