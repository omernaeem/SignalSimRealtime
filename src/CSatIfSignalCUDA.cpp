#include <CSatIfSignalCUDA.h>

//----------------------------------------------------------------------
CSatIfSignalCUDA::CSatIfSignalCUDA(int MsSampleNumber, int SatIfFreq, GnssSystem SatSystem, int SatSignalIndex, unsigned char SatId) : SampleNumber(MsSampleNumber), IfFreq(SatIfFreq), System(SatSystem), SignalIndex(SatSignalIndex), Svid((int)SatId)
{
    // cudaError_t err = cudaMalloc((void**)&SampleArray, SampleNumber * sizeof(cuComplex));
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
    //     SampleArray = nullptr;
    // }

    if (cudaHostAlloc((void**)&SampleArray, SampleNumber * sizeof(cuComplex), cudaHostAllocDefault) != cudaSuccess) {
        throw std::runtime_error("x must not be negative");
    }


    PrnSequence = std::make_unique<PrnGenerate>(System, SignalIndex, Svid);
    SatParam = nullptr;

    if (!PrnSequence->Attribute || !PrnSequence->DataPrn)
        DataLength = PilotLength = 0;
    else
    {
        DataLength = PrnSequence->Attribute->DataPeriod * PrnSequence->Attribute->ChipRate;
        PilotLength = PrnSequence->Attribute->PilotPeriod * PrnSequence->Attribute->ChipRate;
    }
    GlonassHalfCycle = ((IfFreq % 1000) != 0) ? 1 : 0;
    
    // Initialize parameter storage for batch processing
    MaxMilliseconds = 1000;  // Support up to 1 second of data
    ParameterBuffer.reserve(MaxMilliseconds);
    
    // Initialize GPU memory pointers
    d_sampleArray = nullptr;
    d_curChipArray = nullptr;
    d_codeStepArray = nullptr;
    d_curIntPhaseArray = nullptr;
    d_intPhaseStepArray = nullptr;
    d_ampArray = nullptr;
    d_dataSignalArray = nullptr;
    d_pilotSignalArray = nullptr;
    gpuMemoryAllocated = false;
}

CSatIfSignalCUDA::~CSatIfSignalCUDA()
{
    if (SampleArray) {
        cudaFreeHost(SampleArray);
        SampleArray = nullptr;
    }
    FreeGPUMemory();
}

void CSatIfSignalCUDA::InitState(GNSS_TIME CurTime, PSATELLITE_PARAM pSatParam, NavBit* pNavData)
{
    SatParam = pSatParam;
    if (!SatelliteSignal.SetSignalAttribute(System, SignalIndex, pNavData, Svid))
        SatelliteSignal.NavData = nullptr;    // if system/frequency and navigation data not match, set pointer to NULL
    StartCarrierPhase = GetCarrierPhase(SatParam, SignalIndex);
    SignalTime = StartTransmitTime = GetTransmitTime(CurTime, GetTravelTime(SatParam, SignalIndex));
    SatelliteSignal.GetSatelliteSignal(SignalTime, DataSignal, PilotSignal);
    HalfCycleFlag = 0;
}

void CSatIfSignalCUDA::GetIfSample(GNSS_TIME CurTime) {
    // For backward compatibility, collect parameters for this millisecond and generate samples immediately
    CollectParameters(CurTime);
    GenerateSamplesFromParams();
}

void CSatIfSignalCUDA::CollectParameters(GNSS_TIME CurTime) {
	int TransmitMsDiff;
	double CurPhase, PhaseStep, CurChip, CodeDiff, CodeStep;
	unsigned int CurIntPhase;
	int IntPhaseStep;
	const PrnAttribute* CodeAttribute = PrnSequence->Attribute;
	double Amp = pow(10, (SatParam->CN0 - 3000) / 1000.) / sqrt(SampleNumber);

	if (!SatParam)
		return;
		
	SignalTime = StartTransmitTime;
	SatelliteSignal.GetSatelliteSignal(SignalTime, DataSignal, PilotSignal);
	EndCarrierPhase = GetCarrierPhase(SatParam, SignalIndex);
	EndTransmitTime = GetTransmitTime(CurTime, GetTravelTime(SatParam, SignalIndex));

	// calculate start/end signal phase and phase step (actual local signal phase is negative ADR)
	PhaseStep = (StartCarrierPhase - EndCarrierPhase) / SampleNumber;
	PhaseStep += IfFreq / 1000. / SampleNumber;
	CurPhase = StartCarrierPhase - (int)StartCarrierPhase;
	CurPhase = 1 - CurPhase;	// carrier is fractional part of negative of travel time, equvalent to 1 minus positive fractional part
	CurIntPhase = (unsigned int)std::floor(CurPhase * 4294967296.);
	IntPhaseStep = (int)std::round(PhaseStep * 4294967296.);
	StartCarrierPhase = EndCarrierPhase;
	if (GlonassHalfCycle)	// for GLONASS odd number FreqID, nominal IF result in half cycle toggle every 1ms
	{
		CurPhase += HalfCycleFlag ? 0.5 : 0.0;
		HalfCycleFlag = 1 - HalfCycleFlag;
	}

	// get PRN count for each sample
	TransmitMsDiff = EndTransmitTime.MilliSeconds - StartTransmitTime.MilliSeconds;
	if (TransmitMsDiff < 0)
		TransmitMsDiff += 86400000;
	CodeDiff = (TransmitMsDiff + EndTransmitTime.SubMilliSeconds - StartTransmitTime.SubMilliSeconds) * CodeAttribute->ChipRate;
	CodeStep = CodeDiff / SampleNumber;	// code increase between each sample
	CurChip = (StartTransmitTime.MilliSeconds % CodeAttribute->PilotPeriod + StartTransmitTime.SubMilliSeconds) * CodeAttribute->ChipRate;
	StartTransmitTime = EndTransmitTime;

	// Store parameters for this millisecond
	MillisecondParams params;
	params.CurChip = CurChip;
	params.CodeStep = CodeStep;
	params.CurIntPhase = CurIntPhase;
	params.IntPhaseStep = IntPhaseStep;
	params.Amp = Amp;
	params.DataSignal = DataSignal;
	params.PilotSignal = PilotSignal;
	
	ParameterBuffer.push_back(params);
}

void CSatIfSignalCUDA::GenerateSamplesFromParams() {
    if (ParameterBuffer.empty()) {
        return;
    }
    
    // Allocate GPU memory if not already done
    if (!gpuMemoryAllocated) {
        AllocateGPUMemory();
        
        // Copy PRN data to constant memory
        if (PrnSequence && PrnSequence->DataPrn) {
            // Convert int arrays to unsigned char arrays for CUDA kernel
            std::vector<unsigned char> dataPrn(DataLength);
            std::vector<unsigned char> pilotPrn(PilotLength);
            
            for (int i = 0; i < DataLength; i++) {
                dataPrn[i] = (unsigned char)(PrnSequence->DataPrn[i] & 0xFF);
            }
            
            for (int i = 0; i < PilotLength; i++) {
                pilotPrn[i] = PrnSequence->PilotPrn ? (unsigned char)(PrnSequence->PilotPrn[i] & 0xFF) : 0;
            }
            
            copyPrnToConstantMemory(
                dataPrn.data(), DataLength,
                pilotPrn.data(), PilotLength,
                PrnSequence->Attribute ? PrnSequence->Attribute->Attribute : 0
            );
        }
    }
    
    // Copy parameter data to GPU
    CopyParametersToGPU();
    
    int numMs = ParameterBuffer.size();
    
    // Launch CUDA kernel for parallel sample generation
    cudaError_t err = launchGenerateSamplesKernel(
        d_sampleArray,
        d_curChipArray,
        d_codeStepArray,
        d_curIntPhaseArray,
        d_intPhaseStepArray,
        d_ampArray,
        d_dataSignalArray,
        d_pilotSignalArray,
        numMs,
        SampleNumber,
        0  // default stream
    );
    
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Wait for kernel completion
    cudaDeviceSynchronize();
    
    // Copy results back to host - only the last millisecond for compatibility
    // In Phase 3, we'll support copying all milliseconds
    int lastMsOffset = (numMs - 1) * SampleNumber;
    err = cudaMemcpy(SampleArray, d_sampleArray + lastMsOffset, 
                     SampleNumber * sizeof(cuComplex), 
                     cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));
    }
}

void CSatIfSignalCUDA::ClearParameterBuffer() {
    ParameterBuffer.clear();
}

void CSatIfSignalCUDA::AllocateGPUMemory() {
    if (gpuMemoryAllocated) {
        return;
    }
    
    // Allocate GPU memory for sample array and parameter arrays
    size_t sampleArraySize = MaxMilliseconds * SampleNumber * sizeof(cuComplex);
    size_t paramArraySize = MaxMilliseconds * sizeof(double);
    size_t intParamArraySize = MaxMilliseconds * sizeof(int);
    size_t complexParamArraySize = MaxMilliseconds * sizeof(cuComplex);
    
    cudaError_t err;
    
    // Allocate sample array
    err = cudaMalloc((void**)&d_sampleArray, sampleArraySize);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for sample array: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Allocate parameter arrays
    err = cudaMalloc((void**)&d_curChipArray, paramArraySize);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for chip array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    err = cudaMalloc((void**)&d_codeStepArray, paramArraySize);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for code step array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    err = cudaMalloc((void**)&d_curIntPhaseArray, MaxMilliseconds * sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for int phase array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    err = cudaMalloc((void**)&d_intPhaseStepArray, intParamArraySize);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for phase step array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    err = cudaMalloc((void**)&d_ampArray, MaxMilliseconds * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for amplitude array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    err = cudaMalloc((void**)&d_dataSignalArray, complexParamArraySize);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for data signal array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    err = cudaMalloc((void**)&d_pilotSignalArray, complexParamArraySize);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for pilot signal array: %s\n", cudaGetErrorString(err));
        FreeGPUMemory();
        return;
    }
    
    gpuMemoryAllocated = true;
    printf("GPU memory allocated successfully for %d milliseconds\n", MaxMilliseconds);
}

void CSatIfSignalCUDA::FreeGPUMemory() {
    if (!gpuMemoryAllocated) {
        return;
    }
    
    if (d_sampleArray) { cudaFree(d_sampleArray); d_sampleArray = nullptr; }
    if (d_curChipArray) { cudaFree(d_curChipArray); d_curChipArray = nullptr; }
    if (d_codeStepArray) { cudaFree(d_codeStepArray); d_codeStepArray = nullptr; }
    if (d_curIntPhaseArray) { cudaFree(d_curIntPhaseArray); d_curIntPhaseArray = nullptr; }
    if (d_intPhaseStepArray) { cudaFree(d_intPhaseStepArray); d_intPhaseStepArray = nullptr; }
    if (d_ampArray) { cudaFree(d_ampArray); d_ampArray = nullptr; }
    if (d_dataSignalArray) { cudaFree(d_dataSignalArray); d_dataSignalArray = nullptr; }
    if (d_pilotSignalArray) { cudaFree(d_pilotSignalArray); d_pilotSignalArray = nullptr; }
    
    gpuMemoryAllocated = false;
}

void CSatIfSignalCUDA::CopyParametersToGPU() {
    if (!gpuMemoryAllocated || ParameterBuffer.empty()) {
        return;
    }
    
    int numMs = ParameterBuffer.size();
    
    // Prepare host arrays for parameter data
    std::vector<double> h_curChip(numMs);
    std::vector<double> h_codeStep(numMs);
    std::vector<unsigned int> h_curIntPhase(numMs);
    std::vector<int> h_intPhaseStep(numMs);
    std::vector<float> h_amp(numMs);
    std::vector<cuComplex> h_dataSignal(numMs);
    std::vector<cuComplex> h_pilotSignal(numMs);
    
    // Extract parameters from ParameterBuffer
    for (int i = 0; i < numMs; i++) {
        const MillisecondParams& params = ParameterBuffer[i];
        h_curChip[i] = params.CurChip;
        h_codeStep[i] = params.CodeStep;
        h_curIntPhase[i] = params.CurIntPhase;
        h_intPhaseStep[i] = params.IntPhaseStep;
        h_amp[i] = (float)params.Amp;
        h_dataSignal[i] = params.DataSignal;
        h_pilotSignal[i] = params.PilotSignal;
    }
    
    // Copy to GPU
    cudaError_t err;
    
    err = cudaMemcpy(d_curChipArray, h_curChip.data(), numMs * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy chip array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(d_codeStepArray, h_codeStep.data(), numMs * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy code step array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(d_curIntPhaseArray, h_curIntPhase.data(), numMs * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy int phase array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(d_intPhaseStepArray, h_intPhaseStep.data(), numMs * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy phase step array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(d_ampArray, h_amp.data(), numMs * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy amplitude array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(d_dataSignalArray, h_dataSignal.data(), numMs * sizeof(cuComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy data signal array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(d_pilotSignalArray, h_pilotSignal.data(), numMs * sizeof(cuComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy pilot signal array to GPU: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Basic CPU implementations - will be optimized/moved to GPU in Phase 2
cuComplex CSatIfSignalCUDA::GetPrnValue(double& CurChip, double CodeStep) {
    int ChipCount = (int)CurChip;
    int DataChip, PilotChip;
    cuComplex PrnValue = make_cuFloatComplex(0.0f, 0.0f);
    int IsBoc = (PrnSequence->Attribute->Attribute) & PRN_ATTRIBUTE_BOC;
    int IsL2C = (PrnSequence->Attribute->Attribute) & PRN_ATTRIBUTE_TMD;

    if (DataLength == 0)
        return make_cuFloatComplex(0.0f, 0.0f);

    DataChip = ChipCount % DataLength;
    if (IsBoc || IsL2C)
        DataChip >>= 1;
    
    // Convert from cuComplex DataSignal
    if (PrnSequence->DataPrn[DataChip])
        PrnValue = cuCsubf(make_cuFloatComplex(0.0f, 0.0f), DataSignal);
    else
        PrnValue = DataSignal;

    if (PrnSequence->PilotPrn && PilotLength > 0) {
        PilotChip = ChipCount % PilotLength;
        if (IsBoc || IsL2C)
            PilotChip >>= 1;
        
        cuComplex pilotContrib;
        if (PrnSequence->PilotPrn[PilotChip])
            pilotContrib = cuCsubf(make_cuFloatComplex(0.0f, 0.0f), PilotSignal);
        else
            pilotContrib = PilotSignal;
            
        if (IsL2C) {
            if (ChipCount & 1) // L2CL slot
                PrnValue = pilotContrib;
        } else {
            PrnValue = cuCaddf(PrnValue, pilotContrib);
        }
    }
    
    if (IsBoc && (ChipCount & 1)) // second half of BOC code
        PrnValue = cuCsubf(make_cuFloatComplex(0.0f, 0.0f), PrnValue);
    
    CurChip += CodeStep;
    return PrnValue;
}

cuComplex CSatIfSignalCUDA::GetRotateValue(double& CurPhase, double PhaseStep) {
    // Basic rotation using standard trigonometric functions
    // In Phase 2, this will be optimized with FastMath or LUT
    float angle = (float)(CurPhase * 2.0 * M_PI);
    cuComplex Rotate = make_cuFloatComplex(cosf(angle), sinf(angle));
    CurPhase += PhaseStep;
    return Rotate;
}

cuComplex CSatIfSignalCUDA::GetRotateValue(unsigned int& CurPhase, int PhaseStep) {
    // Convert unsigned int phase to float (32-bit phase represents 0-2π)
    float angle = (float)CurPhase * (2.0f * (float)M_PI) / 4294967296.0f;
    cuComplex Rotate = make_cuFloatComplex(cosf(angle), sinf(angle));
    CurPhase += PhaseStep;
    return Rotate;
}