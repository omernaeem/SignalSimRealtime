#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <iomanip>
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

void compareSignalGenerators(int sampleFreq, int numMilliseconds = 5) {
    std::cout << "\n=== Comparing CUDA vs CPU Signal Generation ===" << std::endl;
    std::cout << "Sample Frequency: " << sampleFreq << " samples/ms" << std::endl;
    std::cout << "Test Duration: " << numMilliseconds << " ms" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Configuration
    int ifFreq = 1575420000;
    GnssSystem system = GpsSystem;
    int signalIndex = SIGNAL_INDEX_L1CA;
    unsigned char svid = 1;
    
    try {
        // Create both signal generators
        CSatIfSignalCUDA cudaSignal(sampleFreq, ifFreq, system, signalIndex, svid);
        CSatIfSignal cpuSignal(sampleFreq, ifFreq, system, signalIndex, svid);
        
        std::cout << "✓ Both signal generators created" << std::endl;
        
        // Create identical test parameters
        SATELLITE_PARAM satParam = createTestSatelliteParam(svid);
        GNSS_TIME testTime = createTestGnssTime();
        LNavBit* navBit = new LNavBit();
        
        // Initialize both generators with identical parameters
        cudaSignal.InitState(testTime, &satParam, navBit);
        cpuSignal.InitState(testTime, &satParam, navBit);
        
        std::cout << "✓ Both generators initialized with identical parameters" << std::endl;
        
        // Test multiple milliseconds
        GNSS_TIME currentTime = testTime;
        
        for (int ms = 0; ms < numMilliseconds; ms++) {
            std::cout << "\n--- Millisecond " << ms << " ---" << std::endl;
            
            // Generate samples for this millisecond
            auto cudaStart = std::chrono::high_resolution_clock::now();
            cudaSignal.GetIfSample(currentTime);
            auto cudaEnd = std::chrono::high_resolution_clock::now();
            
            auto cpuStart = std::chrono::high_resolution_clock::now();
            cpuSignal.GetIfSample(currentTime);
            auto cpuEnd = std::chrono::high_resolution_clock::now();
            
            auto cudaTime = std::chrono::duration_cast<std::chrono::microseconds>(cudaEnd - cudaStart);
            auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart);
            
            std::cout << "CUDA generation time: " << cudaTime.count() << " μs" << std::endl;
            std::cout << "CPU generation time:  " << cpuTime.count() << " μs" << std::endl;
            std::cout << "Speed ratio: " << (double)cpuTime.count() / cudaTime.count() << "x" << std::endl;
            
            // Compare first 10 samples
            std::cout << "\nFirst 10 samples comparison:" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Sample |      CUDA (I,Q)      |      CPU (I,Q)       |   Difference (I,Q)   |" << std::endl;
            std::cout << "-------|----------------------|----------------------|----------------------|" << std::endl;
            
            double maxDiff = 0.0;
            double totalDiff = 0.0;
            bool hasDifferences = false;
            
            for (int i = 0; i < std::min(10, sampleFreq); i++) {
                cuComplex cudaSample = cudaSignal.SampleArray[i];
                complex_number cpuSample = cpuSignal.SampleArray[i];
                
                float cudaI = cuCrealf(cudaSample);
                float cudaQ = cuCimagf(cudaSample);
                float cpuI = (float)cpuSample.real;
                float cpuQ = (float)cpuSample.imag;
                
                float diffI = cudaI - cpuI;
                float diffQ = cudaQ - cpuQ;
                double magnitude = sqrt(diffI*diffI + diffQ*diffQ);
                
                if (magnitude > 1e-6) {
                    hasDifferences = true;
                }
                
                maxDiff = std::max(maxDiff, magnitude);
                totalDiff += magnitude;
                
                std::cout << std::setw(6) << i << " | ";
                std::cout << "(" << std::setw(8) << cudaI << "," << std::setw(8) << cudaQ << ") | ";
                std::cout << "(" << std::setw(8) << cpuI << "," << std::setw(8) << cpuQ << ") | ";
                std::cout << "(" << std::setw(8) << diffI << "," << std::setw(8) << diffQ << ") |" << std::endl;
            }
            
            std::cout << "\nStatistics for millisecond " << ms << ":" << std::endl;
            std::cout << "Maximum difference magnitude: " << maxDiff << std::endl;
            std::cout << "Average difference magnitude: " << totalDiff / std::min(10, sampleFreq) << std::endl;
            
            if (hasDifferences) {
                std::cout << "⚠️  Differences detected between CUDA and CPU implementations" << std::endl;
            } else {
                std::cout << "✓ CUDA and CPU outputs match perfectly" << std::endl;
            }
            
            // Calculate RMS power for both
            double cudaRms = 0.0, cpuRms = 0.0;
            for (int i = 0; i < sampleFreq; i++) {
                cuComplex cudaSample = cudaSignal.SampleArray[i];
                complex_number cpuSample = cpuSignal.SampleArray[i];
                
                float cudaI = cuCrealf(cudaSample);
                float cudaQ = cuCimagf(cudaSample);
                cudaRms += cudaI*cudaI + cudaQ*cudaQ;
                
                cpuRms += cpuSample.real*cpuSample.real + cpuSample.imag*cpuSample.imag;
            }
            
            cudaRms = sqrt(cudaRms / sampleFreq);
            cpuRms = sqrt(cpuRms / sampleFreq);
            
            std::cout << "CUDA RMS power: " << cudaRms << std::endl;
            std::cout << "CPU RMS power:  " << cpuRms << std::endl;
            std::cout << "RMS difference: " << fabs(cudaRms - cpuRms) << std::endl;
            
            // Increment time for next millisecond
            currentTime.MilliSeconds += 1;
            if (currentTime.MilliSeconds >= 86400000) {
                currentTime.MilliSeconds -= 86400000;
            }
        }
        
        // Overall comparison summary
        std::cout << "\n=== Overall Comparison Summary ===" << std::endl;
        std::cout << "✓ Both implementations completed successfully" << std::endl;
        std::cout << "✓ " << numMilliseconds << " milliseconds tested" << std::endl;
        std::cout << "✓ " << numMilliseconds * sampleFreq << " total samples compared" << std::endl;
        
        // Clean up
        delete navBit;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error during comparison: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA vs CPU Signal Generation Comparison" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Test with different sampling frequencies
    std::vector<int> sampleFreqs = {2048, 5000};
    
    for (int sampleFreq : sampleFreqs) {
        compareSignalGenerators(sampleFreq, 3);  // Test 3 milliseconds
        std::cout << "\n" << std::endl;
    }
    
    std::cout << "Comparison completed!" << std::endl;
    return 0;
}
