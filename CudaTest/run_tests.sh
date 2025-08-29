#!/bin/bash

# CUDA Test Runner Script
# Builds and runs CUDA signal processing tests

echo "======================================="
echo "  CUDA Signal Processing Test Runner"
echo "======================================="

# Change to build directory
cd "$(dirname "$0")/../build" || exit 1

# Build the project
echo "Building CUDA test executables..."
if ! make CudaTest AdvancedCudaGuide; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Check if user wants to run a specific test
if [ "$1" = "basic" ]; then
    echo "Running basic CUDA tests..."
    ./CudaTest/CudaTest
elif [ "$1" = "advanced" ]; then
    echo "Running advanced CUDA guide..."
    ./CudaTest/AdvancedCudaGuide
elif [ "$1" = "both" ] || [ "$1" = "" ]; then
    echo "Running basic CUDA tests..."
    echo "======================================="
    ./CudaTest/CudaTest
    
    echo ""
    echo "======================================="
    echo "Running advanced CUDA guide..."
    echo "======================================="
    ./CudaTest/AdvancedCudaGuide
else
    echo "Usage: $0 [basic|advanced|both]"
    echo "  basic    - Run basic CUDA functionality tests"
    echo "  advanced - Run advanced CUDA guide and examples"
    echo "  both     - Run both tests (default)"
    exit 1
fi

echo ""
echo "✅ All tests completed!"
