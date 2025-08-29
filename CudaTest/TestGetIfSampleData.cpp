#include <stdio.h>
#include <iostream>
#include <iomanip>
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

int main() {
    std::cout << "=== Testing GetIfSample Data Generation ===" << std::endl;
    
    // Test configuration for GPS L1 C/A
    int sampleFreq = 2048;  // samples per ms
    int ifFreq = 1575420000;  // Hz
    GnssSystem system = GpsSystem;
    int signalIndex = SIGNAL_INDEX_L1CA;
    unsigned char svid = 1;
    
    try {
        // Create CUDA signal generator
        CSatIfSignalCUDA cudaSignal(sampleFreq, ifFreq, system, signalIndex, svid);
        std::cout << "✓ CUDA signal generator created" << std::endl;
        
        // Create test satellite parameters
        SATELLITE_PARAM satParam = createTestSatelliteParam(svid);
        std::cout << "✓ Test satellite parameters created" << std::endl;
        
        // Create test time
        GNSS_TIME testTime = createTestGnssTime();
        std::cout << "✓ Test GNSS time created" << std::endl;
        
        // Create navigation data
        LNavBit* navBit = new LNavBit();
        std::cout << "✓ L-NAV navigation bit generator created" << std::endl;
        
        // Initialize the signal state
        std::cout << "✓ Calling InitState..." << std::endl;
        cudaSignal.InitState(testTime, &satParam, navBit);
        std::cout << "✓ InitState completed" << std::endl;
        
        std::cout << "\n=== Calling GetIfSample to verify data generation ===" << std::endl;
        
        // Add some debug prints before calling GetIfSample to show input parameters
        std::cout << "Input parameters to GetIfSample:" << std::endl;
        std::cout << "  Sample Frequency: " << sampleFreq << " samples/ms" << std::endl;
        std::cout << "  IF Frequency: " << ifFreq << " Hz" << std::endl;
        std::cout << "  Satellite CN0: " << satParam.CN0/100.0 << " dB-Hz" << std::endl;
        std::cout << "  Travel Time: " << satParam.TravelTime*1000 << " ms" << std::endl;
        std::cout << "  Test Time Week: " << testTime.Week << std::endl;
        std::cout << "  Test Time MS: " << testTime.MilliSeconds << std::endl;
        
        // Call GetIfSample
        cudaSignal.GetIfSample(testTime);
        std::cout << "✓ GetIfSample completed!" << std::endl;
        
        // Check if sample array has data
        if (cudaSignal.SampleArray != nullptr) {
            std::cout << "\n✓ Sample array is accessible" << std::endl;
            
            // Print first few samples to verify generation
            std::cout << "\nFirst 10 samples generated:" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            for (int i = 0; i < 10 && i < sampleFreq; i++) {
                cuComplex sample = cudaSignal.SampleArray[i];
                std::cout << "  Sample " << std::setw(2) << i << ": (" 
                          << std::setw(10) << cuCrealf(sample) << ", " 
                          << std::setw(10) << cuCimagf(sample) << ")" << std::endl;
            }
            
            // Check for non-zero samples
            bool hasNonZeroSamples = false;
            double maxMagnitude = 0.0;
            for (int i = 0; i < sampleFreq; i++) {
                cuComplex sample = cudaSignal.SampleArray[i];
                double magnitude = sqrt(cuCrealf(sample)*cuCrealf(sample) + cuCimagf(sample)*cuCimagf(sample));
                if (magnitude > maxMagnitude) {
                    maxMagnitude = magnitude;
                }
                if (magnitude > 1e-10) {
                    hasNonZeroSamples = true;
                }
            }
            
            std::cout << "\nSignal analysis:" << std::endl;
            std::cout << "  Non-zero samples detected: " << (hasNonZeroSamples ? "Yes" : "No") << std::endl;
            std::cout << "  Maximum sample magnitude: " << maxMagnitude << std::endl;
            
            if (hasNonZeroSamples) {
                std::cout << "✓ SUCCESS: Signal generation is working!" << std::endl;
            } else {
                std::cout << "⚠️  WARNING: All samples are zero - check signal parameters" << std::endl;
            }
        } else {
            std::cout << "✗ ERROR: Sample array is null" << std::endl;
        }
        
        // Clean up
        delete navBit;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}
