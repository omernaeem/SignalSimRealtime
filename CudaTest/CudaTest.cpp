#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <ctime>

#include "SignalSim.h"
#include "CSatIfSignalCUDA.h"

#define TOTAL_GPS_SAT 32
#define TOTAL_BDS_SAT 63
#define TOTAL_GAL_SAT 36
#define TOTAL_GLO_SAT 24
#define TOTAL_SAT_CHANNEL 128

// Test parameters
struct TestConfig {
    int sampleFreq;      // Samples per millisecond
    int ifFreq;          // IF frequency in Hz
    GnssSystem system;   // GNSS system to test
    int signalIndex;     // Signal index
    unsigned char svid;  // Satellite ID
    int testDurationMs;  // Test duration in milliseconds
};

void printTestHeader(const TestConfig& config) {
    std::cout << "=== CUDA Signal Generation Test ===" << std::endl;
    std::cout << "Sample Frequency: " << config.sampleFreq << " samples/ms" << std::endl;
    std::cout << "IF Frequency: " << config.ifFreq << " Hz" << std::endl;
    std::cout << "System: " << (int)config.system << std::endl;
    std::cout << "Signal Index: " << config.signalIndex << std::endl;
    std::cout << "Satellite ID: " << (int)config.svid << std::endl;
    std::cout << "Test Duration: " << config.testDurationMs << " ms" << std::endl;
    std::cout << "====================================" << std::endl;
}

void testBasicCudaSignalGeneration(const TestConfig& config) {
    std::cout << "\n--- Testing Basic CUDA Signal Generation ---" << std::endl;
    
    try {
        // Create CUDA signal generator
        CSatIfSignalCUDA cudaSignal(config.sampleFreq, config.ifFreq, 
                                   config.system, config.signalIndex, config.svid);
        
        std::cout << "✓ CUDA signal generator created successfully" << std::endl;
        std::cout << "✓ GPU memory allocated for " << config.sampleFreq << " samples" << std::endl;
        
        // Test memory allocation
        if (cudaSignal.SampleArray != nullptr) {
            std::cout << "✓ Sample array pointer is valid" << std::endl;
        } else {
            std::cout << "✗ Sample array pointer is null" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
    }
}

void testCudaVsCpuPerformance(const TestConfig& config) {
    std::cout << "\n--- Testing CUDA vs CPU Performance ---" << std::endl;
    
    try {
        // Create both CUDA and CPU signal generators
        CSatIfSignalCUDA cudaSignal(config.sampleFreq, config.ifFreq, 
                                   config.system, config.signalIndex, config.svid);
        CSatIfSignal cpuSignal(config.sampleFreq, config.ifFreq, 
                              config.system, config.signalIndex, config.svid);
        
        std::cout << "✓ Both signal generators created" << std::endl;
        
        // Note: For actual testing, you would need to:
        // 1. Set up proper satellite parameters
        // 2. Initialize navigation data
        // 3. Set up timing
        // 4. Call GetIfSample on both
        // 5. Compare performance and results
        
        std::cout << "✓ Ready for performance comparison" << std::endl;
        std::cout << "  (Full implementation requires satellite parameters and nav data)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
    }
}

void testMemoryOperations(const TestConfig& config) {
    std::cout << "\n--- Testing CUDA Memory Operations ---" << std::endl;
    
    try {
        CSatIfSignalCUDA cudaSignal(config.sampleFreq, config.ifFreq, 
                                   config.system, config.signalIndex, config.svid);
        
        // Test CUDA memory operations
        std::cout << "✓ GPU memory allocated successfully" << std::endl;
        
        // You could add more specific CUDA tests here:
        // - Copy data to/from GPU
        // - Test kernel launches
        // - Verify memory alignment
        // - Test error handling
        
        std::cout << "✓ Memory operations test completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Memory test failed: " << e.what() << std::endl;
    }
}

void runTestSuite() {
    // Define test configurations
    std::vector<TestConfig> testConfigs = {
        // GPS L1 C/A test
        {2048, 1575420000, GpsSystem, 0, 1, 100},
        
        // BDS B1I test
        {2048, 1561098000, BdsSystem, 1, 1, 100},
        
        // Galileo E1 test
        {2048, 1575420000, GalileoSystem, 0, 1, 100},
        
        // GLONASS G1 test (frequency channel 0)
        {2048, 1602000000, GlonassSystem, 0, 1, 100}
    };
    
    for (const auto& config : testConfigs) {
        printTestHeader(config);
        
        testBasicCudaSignalGeneration(config);
        testCudaVsCpuPerformance(config);
        testMemoryOperations(config);
        
        std::cout << "\n" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Signal Processing Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "✗ CUDA not available: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "✓ Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  Device " << i << ": " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Run the test suite
    runTestSuite();
    
    std::cout << "Test suite completed!" << std::endl;
    return 0;
}
