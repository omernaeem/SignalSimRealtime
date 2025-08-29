# CUDA Signal Processing Test Framework

This directory contains a dedicated test framework for CUDA-based GNSS signal processing, specifically designed to test and develop the `CSatIfSignalCUDA` class and its `GetIfSample` function.

## Overview

The CudaTest framework provides a separate executable that allows you to:
- Test CUDA functionality independently from the main IFdataGen application
- Develop and debug CUDA kernels for signal generation
- Compare CUDA vs CPU performance
- Access all source files and headers from the main project

## Files

- **CudaTest.cpp** - Basic CUDA functionality tests
- **AdvancedCudaGuide.cpp** - Comprehensive guide and advanced testing
- **CMakeLists.txt** - Build configuration with proper CUDA support

## Executables

### CudaTest
Basic test suite that validates:
- CUDA device availability
- Memory allocation on GPU
- Signal generator creation for different GNSS systems
- Basic functionality verification

```bash
./build/CudaTest/CudaTest
```

### AdvancedCudaGuide
Comprehensive guide that demonstrates:
- How to use the GetIfSample function
- Key data structures (SATELLITE_PARAM, GNSS_TIME, NavBit)
- CUDA vs CPU implementation comparison
- Performance optimization strategies

```bash
./build/CudaTest/AdvancedCudaGuide
```

## Building

The CudaTest framework is integrated into the main CMake build system:

```bash
cd build
cmake ..
make CudaTest                # Build basic test
make AdvancedCudaGuide      # Build advanced guide
make                        # Build all targets
```

## Usage Pattern

### Basic Testing
```cpp
// Create CUDA signal generator
CSatIfSignalCUDA cudaSignal(sampleFreq, ifFreq, system, signalIndex, svid);

// Test basic functionality
if (cudaSignal.SampleArray != nullptr) {
    std::cout << "✓ GPU memory allocated successfully" << std::endl;
}
```

### Complete Workflow
```cpp
// 1. Create signal generator
CSatIfSignalCUDA cudaSignal(2048, 1575420000, GpsSystem, 0, 1);

// 2. Set up satellite parameters (requires ephemeris data)
SATELLITE_PARAM satParam;
// ... populate satellite position, velocity, clock offset

// 3. Set up navigation data
NavBit navData;
// ... load navigation message

// 4. Initialize state
GNSS_TIME currentTime = {2100, 345000, 0.5};
cudaSignal.InitState(currentTime, &satParam, &navData);

// 5. Generate samples
cudaSignal.GetIfSample(currentTime);

// 6. Access results from cudaSignal.SampleArray
```

## Key Features

### CUDA Support
- Proper CUDA toolkit integration
- GPU memory management
- Error handling for CUDA operations
- Support for multiple GPU architectures

### GNSS Systems
Supports all major GNSS systems:
- **GPS** (L1 C/A, L1C, L2C, L5)
- **BeiDou** (B1C, B1I, B2I, B3I, B2a, B2b)
- **Galileo** (E1, E5a, E5b, E6)
- **GLONASS** (G1, G2)

### Testing Scenarios
- Single satellite signal generation
- Multiple satellite scenarios
- Different signal types and frequencies
- Performance benchmarking

## Development Notes

### Implementing CUDA Kernels
The framework expects CUDA kernels to be implemented in `../src/kernel.cu`. Key functions to implement:

```cuda
__global__ void generateSignalSamples(
    cuComplex* samples,
    int sampleCount,
    double codeStep,
    double phaseStep,
    // ... other parameters
);
```

### Performance Optimization
- Use GPU memory coalescing
- Optimize thread block sizes
- Minimize host-device memory transfers
- Consider shared memory usage

### Error Handling
Always check CUDA errors:
```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
}
```

## Requirements

- CUDA Toolkit 12.4 or later
- CMake 3.18 or later
- C++17 compatible compiler
- NVIDIA GPU with Compute Capability 7.5 or higher

## Troubleshooting

### Common Issues
1. **CUDA not found**: Ensure CUDA toolkit is properly installed
2. **Compute capability mismatch**: Update CMAKE_CUDA_ARCHITECTURES in CMakeLists.txt
3. **Memory allocation failures**: Check GPU memory availability
4. **Linking errors**: Verify CUDA libraries are properly linked

### Debug Output
Enable verbose CUDA error checking by setting:
```bash
export CUDA_LAUNCH_BLOCKING=1
```

## Integration with Main Project

This test framework:
- Shares all source files and headers with the main project
- Uses the same build system and dependencies
- Excludes main() functions from other executables
- Maintains consistency with IFdataGen implementation

## Next Steps

1. **Implement CUDA kernels** in kernel.cu
2. **Add performance profiling** tools
3. **Create unit tests** for individual functions
4. **Add memory transfer optimization**
5. **Implement batch processing** for multiple satellites

## Example Output

```
CUDA Signal Processing Test Suite
=================================
✓ Found 1 CUDA device(s)
  Device 0: NVIDIA GeForce RTX 3090 (Compute 8.6)

=== CUDA Signal Generation Test ===
✓ CUDA signal generator created successfully
✓ GPU memory allocated for 2048 samples
✓ Sample array pointer is valid
✓ Ready for performance comparison
✓ Memory operations test completed
```

This framework provides a solid foundation for developing and testing CUDA-accelerated GNSS signal processing while maintaining full access to the existing codebase.
