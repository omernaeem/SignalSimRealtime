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
#include "LNavBit.h"
#include "SatelliteParam.h"
#include "Coordinate.h"
#include "GnssTime.h"

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

// Create a dummy satellite parameter structure for testing
SATELLITE_PARAM createTestSatelliteParam(int svid) {
    SATELLITE_PARAM satParam;
    memset(&satParam, 0, sizeof(SATELLITE_PARAM));
    
    // Basic satellite info
    satParam.system = GpsSystem;
    satParam.svid = svid;
    satParam.FreqID = 0;  // GPS doesn't use frequency ID
    satParam.CN0 = 4500;  // 45 dB-Hz typical GPS signal strength
    satParam.PosTimeTag = 0;
    
    // Simulated satellite position (approximately 20000 km above Earth)
    satParam.PosVel.x = 15000000.0;  // meters
    satParam.PosVel.y = 10000000.0;  // meters  
    satParam.PosVel.z = 12000000.0;  // meters
    satParam.PosVel.vx = 3000.0;     // m/s
    satParam.PosVel.vy = 2000.0;     // m/s
    satParam.PosVel.vz = -1000.0;    // m/s
    
    // Typical values for GPS
    satParam.TravelTime = 0.067;     // ~67ms typical GPS travel time
    satParam.IonoDelay = 2.0;        // 2 meters ionospheric delay
    satParam.Elevation = 45.0 * M_PI / 180.0;  // 45 degrees elevation
    satParam.Azimuth = 90.0 * M_PI / 180.0;    // 90 degrees azimuth
    satParam.RelativeSpeed = -500.0;  // m/s relative velocity
    
    // Group delays (all in seconds)
    satParam.GroupDelay[SIGNAL_INDEX_L1CA] = 1.0e-9;  // 1 ns typical
    satParam.GroupDelay[SIGNAL_INDEX_L1C] = 1.2e-9;
    satParam.GroupDelay[SIGNAL_INDEX_L2C] = 0.8e-9;
    satParam.GroupDelay[SIGNAL_INDEX_L5] = 1.1e-9;
    
    // Line of sight vector (unit vector pointing from receiver to satellite)
    double range = sqrt(satParam.PosVel.x*satParam.PosVel.x + 
                       satParam.PosVel.y*satParam.PosVel.y + 
                       satParam.PosVel.z*satParam.PosVel.z);
    satParam.LosVector[0] = satParam.PosVel.x / range;
    satParam.LosVector[1] = satParam.PosVel.y / range;
    satParam.LosVector[2] = satParam.PosVel.z / range;
    
    return satParam;
}

// Create a test GNSS time structure
GNSS_TIME createTestGnssTime() {
    GNSS_TIME gnssTime;
    
    // Set to a typical GPS time (Week 2200, 100 seconds into the week)
    gnssTime.Week = 2200;
    gnssTime.MilliSeconds = 100000;  // 100 seconds = 100,000 ms
    gnssTime.SubMilliSeconds = 0.5;   // 500 ms additional
    
    return gnssTime;
}

void testParameterCollection(const TestConfig& config, int maxMilliseconds) {
    std::cout << "\n--- Testing Parameter Collection for " << maxMilliseconds << "ms ---" << std::endl;
    
    try {
        // Create CUDA signal generator
        CSatIfSignalCUDA cudaSignal(config.sampleFreq, config.ifFreq, 
                                   config.system, config.signalIndex, config.svid);
        
        std::cout << "✓ CUDA signal generator created" << std::endl;
        
        // Create test satellite parameters
        SATELLITE_PARAM satParam = createTestSatelliteParam(config.svid);
        GNSS_TIME testTime = createTestGnssTime();
        
        // Create navigation data
        LNavBit* navBit = nullptr;
        if (config.system == GpsSystem && config.signalIndex == SIGNAL_INDEX_L1CA) {
            navBit = new LNavBit();
        }
        
        // Initialize the signal state
        cudaSignal.InitState(testTime, &satParam, navBit);
        std::cout << "✓ Signal state initialized" << std::endl;
        
        // Clear any existing parameters
        cudaSignal.ClearParameterBuffer();
        
        // Test parameter collection timing
        GNSS_TIME currentTime = testTime;
        auto totalStart = std::chrono::high_resolution_clock::now();
        
        std::cout << "✓ Collecting parameters for " << maxMilliseconds << " milliseconds..." << std::endl;
        
        // Progress reporting every 100ms
        for (int ms = 0; ms < maxMilliseconds; ms++) {
            if (ms % 100 == 0 || ms < 10) {
                std::cout << "  Processing ms " << ms << "..." << std::endl;
            }
            
            auto msStart = std::chrono::high_resolution_clock::now();
            
            // Collect parameters for this millisecond
            cudaSignal.CollectParameters(currentTime);
            
            auto msEnd = std::chrono::high_resolution_clock::now();
            auto msDuration = std::chrono::duration_cast<std::chrono::microseconds>(msEnd - msStart);
            
            if (ms < 10) {
                std::cout << "    ✓ ms " << ms << " parameter collection: " << msDuration.count() << " μs" << std::endl;
            }
            
            // Increment time by 1 millisecond
            currentTime.MilliSeconds += 1;
            if (currentTime.MilliSeconds >= 86400000) {
                currentTime.MilliSeconds -= 86400000;
            }
        }
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
        
        std::cout << "✓ Parameter collection completed!" << std::endl;
        std::cout << "✓ Total time: " << totalDuration.count() << " ms" << std::endl;
        std::cout << "✓ Average time per ms: " << (double)totalDuration.count() / maxMilliseconds << " ms" << std::endl;
        std::cout << "✓ Parameters collected for " << maxMilliseconds << " milliseconds" << std::endl;
        std::cout << "✓ Total samples that will be generated: " << maxMilliseconds * config.sampleFreq << std::endl;
        
        // Test sample generation from first few parameters
        std::cout << "\n✓ Testing sample generation from collected parameters..." << std::endl;
        auto sampleStart = std::chrono::high_resolution_clock::now();
        
        cudaSignal.GenerateSamplesFromParams();  // Generate samples for the last ms
        
        auto sampleEnd = std::chrono::high_resolution_clock::now();
        auto sampleDuration = std::chrono::duration_cast<std::chrono::microseconds>(sampleEnd - sampleStart);
        
        std::cout << "✓ Sample generation time: " << sampleDuration.count() << " μs" << std::endl;
        
        // Check generated samples
        if (cudaSignal.SampleArray != nullptr) {
            bool hasNonZeroSamples = false;
            double rms = 0.0;
            
            for (int i = 0; i < config.sampleFreq; i++) {
                cuComplex sample = cudaSignal.SampleArray[i];
                float real = cuCrealf(sample);
                float imag = cuCimagf(sample);
                
                if (real != 0.0f || imag != 0.0f) {
                    hasNonZeroSamples = true;
                }
                rms += real * real + imag * imag;
            }
            
            rms = sqrt(rms / config.sampleFreq);
            
            if (hasNonZeroSamples) {
                std::cout << "✓ Non-zero samples generated successfully" << std::endl;
                std::cout << "✓ RMS power: " << rms << std::endl;
                
                // Show first few samples
                std::cout << "✓ First 3 samples: ";
                for (int i = 0; i < 3; i++) {
                    cuComplex sample = cudaSignal.SampleArray[i];
                    std::cout << "(" << cuCrealf(sample) << "," << cuCimagf(sample) << ") ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "⚠️  All samples are zero" << std::endl;
            }
        }
        
        // Memory usage estimation
        size_t paramMemory = maxMilliseconds * sizeof(MillisecondParams);
        size_t sampleMemory = (size_t)maxMilliseconds * config.sampleFreq * sizeof(cuComplex);
        
        std::cout << "\n📊 Memory Usage Analysis:" << std::endl;
        std::cout << "  Parameter storage: " << paramMemory / 1024 << " KB (" << paramMemory << " bytes)" << std::endl;
        std::cout << "  Sample storage: " << sampleMemory / (1024*1024) << " MB (" << sampleMemory << " bytes)" << std::endl;
        std::cout << "  Memory efficiency: " << (double)paramMemory / sampleMemory * 100.0 << "% (parameters vs samples)" << std::endl;
        
        // Clean up
        cudaSignal.ClearParameterBuffer();
        if (navBit) {
            delete navBit;
        }
        
        std::cout << "✓ Parameter collection test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
    }
}

void runTestSuite() {
    // Define test configurations - focusing on GPS L1CA for simplicity
    std::vector<TestConfig> testConfigs = {
        {5000, 1575420000, GpsSystem, SIGNAL_INDEX_L1CA, 1, 1000},  // 5 MHz (target)
    };
    
    for (const auto& config : testConfigs) {
        printTestHeader(config);

        testParameterCollection(config, 500);  // 500ms test        
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
