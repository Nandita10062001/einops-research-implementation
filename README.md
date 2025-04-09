# Einops Modified Implementation

## Overview
This project implements a modified version of the einops tensor manipulation library, focusing on numpy array operations. The implementation provides a flexible way to manipulate tensor dimensions using a simple string-based syntax inspired by Einstein notation.

## Table of Contents
- [Implementation Approach](#implementation-approach)
- [Design Decisions](#design-decisions)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Implementation Details](#implementation-details)
- [Examples](#examples)
- [Performance Considerations](#performance-considerations)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

## Implementation Approach
I approached this implementation with a focus on readability, maintainability, and performance. The core functionality is built around the `rearrange` function, which parses pattern strings and performs tensor manipulations. Here's a high-level overview of my approach:

1. **Pattern Parsing**: Implemented a robust parser with LRU caching for efficient pattern string interpretation
2. **Error Handling**: Created custom exception classes for detailed error messages
3. **Memory Management**: Added memory usage estimation to prevent OOM errors
4. **Testing**: Developed comprehensive test suite covering basic operations to edge cases

### Implementation Flowchart
[Implementation Flowchart]
```
Input Pattern String
       ↓
Pattern Parsing (LRU Cached)
       ↓
Validate Input Shape
       ↓
Memory Usage Check
       ↓
Apply Operations:
  ├→ Basic Transpose
  ├→ Split Axes
  ├→ Merge Axes
  ├→ Handle Ellipsis
  └→ Repeat Axes
       ↓
Return Result
```

## Design Decisions

### 1. Pattern String Format
- Used space-separated dimensions for readability
- Implemented parentheses for grouping dimensions
- Added ellipsis (...) support for batch dimensions
- Example: `'(h w) c -> h w c'`

### 2. Error Handling
Created specific exception classes:
```python
class EinopsError(Exception): pass
class PatternError(EinopsError): pass
class ShapeError(EinopsError): pass
class MemoryError(EinopsError): pass
class DimensionError(EinopsError): pass
```

### 3. Performance Optimizations
- LRU caching for pattern parsing
- Memory usage estimation
- Minimal intermediate tensor operations

## Features
- Basic tensor operations (reshape, transpose)
- Axis splitting and merging
- Axis repetition
- Batch dimension handling
- Memory usage control
- Comprehensive error checking
- Data type preservation

## Installation

### Required Libraries
```bash
numpy==1.24.0  # For array operations
pytest==7.4.0  # For running tests
psutil==5.9.0  # For memory monitoring
```

To install the required dependencies:

1. Create a virtual environment (recommended):
```bash
python -m venv einops_env
source einops_env/bin/activate  # On Windows: einops_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install numpy pytest psutil
```

## Usage

Here are examples from our implementation demonstrating various operations:

### 1. Basic Transpose Operation
```python
import numpy as np
from einops_modified import rearrange

# Create a 3x4 array
x = np.random.rand(3, 4)
result = rearrange(x, 'h w -> w h')
# Result shape will be (4, 3)
```

### 2. Split an Axis
```python
# Create a 12x10 array
x = np.random.rand(12, 10)
# Split first dimension into h=3 and w=4
result = rearrange(x, '(h w) c -> h w c', h=3)
# Result shape will be (3, 4, 10)
```

### 3. Merge Axes
```python
# Create a 3x4x5 array
x = np.random.rand(3, 4, 5)
# Merge first two dimensions
result = rearrange(x, 'a b c -> (a b) c')
# Result shape will be (12, 5)
```

### 4. Handle Batch Dimensions
```python
# Create a 2x3x4x5 array
x = np.random.rand(2, 3, 4, 5)
# Merge h and w while preserving batch dimensions
result = rearrange(x, '... h w -> ... (h w)')
# Result shape will be (2, 3, 20)
```

### 5. Complex Transformations
```python
# Create a 24x10x5 array
x = np.random.rand(24, 10, 5)
# Multiple splits and merges
result = rearrange(x, '(h w) c d -> h w (c d)', h=4)
# Result shape will be (4, 6, 50)
```

### 6. Working with Different Data Types
```python
# Float32 arrays
x_float32 = np.random.rand(3, 4).astype(np.float32)
result_float32 = rearrange(x_float32, 'h w -> w h')
# Preserves float32 dtype

# Integer arrays
x_int64 = np.random.randint(0, 100, size=(3, 4), dtype=np.int64)
result_int64 = rearrange(x_int64, 'h w -> w h')
# Preserves int64 dtype

# Boolean arrays
x_bool = np.random.choice([True, False], size=(3, 4))
result_bool = rearrange(x_bool, 'h w -> w h')
# Preserves boolean dtype
```

### Error Handling Examples
```python
# Invalid pattern
try:
    result = rearrange(x, 'invalid pattern')
except PatternError as e:
    print(f"Invalid pattern error: {e}")

# Shape mismatch
try:
    result = rearrange(x, '(h w) c -> h w c', h=5)
except ShapeError as e:
    print(f"Shape mismatch error: {e}")

# Memory limit exceeded
try:
    x_large = np.random.rand(1000, 1000, 1000)
    result = rearrange(x_large, 'a b c -> (a b) c')
except MemoryError as e:
    print(f"Memory limit error: {e}")
```

## Testing
The implementation includes comprehensive tests covering:

1. Basic Operations
   - Transpose operations
   - Axis splitting
   - Axis merging
   - Repetition

2. Complex Transformations
   - Multiple operations
   - Batch dimensions
   - Nested operations

3. Error Cases
   - Invalid patterns
   - Shape mismatches
   - Memory limits
   - Nested parentheses

4. Edge Cases
   - Empty tensors
   - Single dimensions
   - Zero dimensions
   - Large dimensions

5. Performance Tests
   - Execution time
   - Memory usage

6. Data Type Tests
   - float32
   - int64
   - boolean

### Test Output Examples
[Test Output Screenshots]

## Implementation Details

### Core Components
1. **Pattern Parser**
   - Handles input/output patterns
   - Validates syntax
   - Extracts dimension information

2. **Shape Validator**
   - Checks dimension compatibility
   - Validates axis lengths
   - Handles ellipsis expansion

3. **Memory Manager**
   - Estimates memory requirements
   - Prevents excessive memory usage
   - Manages tensor operations efficiently

## Examples
The implementation includes 25 comprehensive examples demonstrating various features:

1. Basic transpose operations
2. Axis splitting and merging
3. Complex pattern handling
4. Error cases
5. Edge cases
6. Performance tests
7. Data type preservation

### Example Output
[Example Output Screenshots]

## Performance Considerations
- Pattern parsing is cached using LRU cache
- Memory usage is estimated before operations
- Intermediate tensor operations are minimized
- Large tensor operations are protected against OOM

## Known Limitations
1. Maximum memory usage is set to 100MB for testing
2. Nested parentheses are not supported
3. Limited to numpy arrays
4. Single ellipsis per pattern

## Future Improvements
1. Support for nested parentheses
2. Dynamic memory limit adjustment
3. Additional optimization for large tensors
4. Support for more complex patterns
5. Integration with other array libraries



## Author
Nandita Nandakumar,
Machine Learning Engineer, Atmos


---
