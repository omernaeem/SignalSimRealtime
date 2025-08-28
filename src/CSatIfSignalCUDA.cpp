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
}

CSatIfSignalCUDA::~CSatIfSignalCUDA()
{
    if (SampleArray) {
        cudaFreeHost(SampleArray);
        SampleArray = nullptr;
    }
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

    int i, TransmitMsDiff;
    double CurPhase, PhaseStep, CurChip, CodeDiff, CodeStep;
    unsigned int CurIntPhase;
    int IntPhaseStep;
    const PrnAttribute* CodeAttribute = PrnSequence->Attribute;
    // complex_number IfSample;
    double Amp = pow(10, (SatParam->CN0 - 3000) / 1000.) / sqrt(SampleNumber);

    if (!SatParam)
        return;
    SignalTime = StartTransmitTime;
    SatelliteSignal.GetSatelliteSignal(SignalTime, DataSignal, PilotSignal);
    EndCarrierPhase = GetCarrierPhase(SatParam, SignalIndex);
    EndTransmitTime = GetTransmitTime(CurTime, GetTravelTime(SatParam, SignalIndex));
    StartCarrierPhase = EndCarrierPhase;
    if (GlonassHalfCycle)    // for GLONASS odd number FreqID, nominal IF result in half cycle toggle every 1ms
    {
        CurPhase += HalfCycleFlag ? 0.5 : 0.0;
        HalfCycleFlag = 1 - HalfCycleFlag;
    }

    // get PRN count for each sample
    TransmitMsDiff = EndTransmitTime.MilliSeconds - StartTransmitTime.MilliSeconds;
    if (TransmitMsDiff < 0)
        TransmitMsDiff += 86400000;
    CodeDiff = (TransmitMsDiff + EndTransmitTime.SubMilliSeconds - StartTransmitTime.SubMilliSeconds) * CodeAttribute->ChipRate;
    CodeStep = CodeDiff / SampleNumber;    // code increase between each sample
    CurChip = (StartTransmitTime.MilliSeconds % CodeAttribute->PilotPeriod + StartTransmitTime.SubMilliSeconds) * CodeAttribute->ChipRate;
    StartTransmitTime = EndTransmitTime;

    // FastMath::InitializeLUT();
    // for (i = 0; i < SampleNumber; i++)
    // {
    //     // SampleArray[i] = GetPrnValue(CurChip, CodeStep) * GetRotateValue(CurPhase, PhaseStep) * Amp;
    //     SampleArray[i] = GetPrnValue(CurChip, CodeStep) * GetRotateValue(CurIntPhase, IntPhaseStep) * Amp;
    // }
}