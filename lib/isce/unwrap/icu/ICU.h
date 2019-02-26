#ifndef ISCE_UNWRAP_ICU_ICU_H
#define ISCE_UNWRAP_ICU_ICU_H

#include <array> // std::array
#include <complex> // std::complex
#include <cstddef> // size_t
#include <cstdint> // uint8_t

#include <isce/io/Raster.h> // isce::io::Raster

#include "LabelMap.h" // LabelMap

namespace isce { namespace unwrap { namespace icu {

// 2-D index type
typedef std::array<size_t, 2> idx2_t;

// 2-D offset type
typedef std::array<int, 2> offset2_t;

class ICU
{
public:
    /** Constructor */
    ICU() = default;

    /** Destructor */
    ~ICU() = default;

    /** Get tile buffer length. */
    size_t numBufLines() const;
    /** Set tile buffer length (default: 3700). */
    void numBufLines(const size_t);

    /** Get lines of overlap between tiles. */
    size_t numOverlapLines() const;
    /** Set lines of overlap between tiles (default: 200). */
    void numOverlapLines(const size_t);

    /** Get phase gradient neutrons flag. */
    bool usePhaseGradNeut() const;
    /** Set phase gradient neutrons flag (default: false). */
    void usePhaseGradNeut(const bool);

    /** Get intensity neutrons flag. */
    bool useIntensityNeut() const;
    /** Set intensity neutrons flag (default: false). */
    void useIntensityNeut(const bool);

    /** Get window size for phase gradient calculation. */
    int phaseGradWinSize() const;
    /** Set window size for phase gradient calculation (default: 5). */
    void phaseGradWinSize(const int);

    /** Get range phase gradient threshold for neutron generation (rad/sample). */
    float neutPhaseGradThr() const;
    /** Set range phase gradient threshold for neutron generation (rad/sample) (default: 3.0). */
    void neutPhaseGradThr(const float);

    /** Get intensity variance threshold for neutron generation (stddevs from mean). */
    float neutIntensityThr() const;
    /** Set intensity variance threshold for neutron generation (stddevs from mean) (default: 8.0). */
    void neutIntensityThr(const float);

    /** Get correlation threshold for neutron generation. */
    float neutCorrThr() const;
    /** Set correlation threshold for neutron generation (default: 0.8). */
    void neutCorrThr(const float);

    /** Get number of tree growing iterations. */
    int numTrees() const;
    /** Set number of tree growing iterations (default: 7). */
    void numTrees(const int);

    /** Get max branch cut length. */
    int maxBranchLen() const;
    /** Set max branch cut length (default: 64). */
    void maxBranchLen(const int);

    /** Get ratio of x:y pixel spacing (for measuring branch cut length). */
    float ratioDxDy() const;
    /** Set ratio of x:y pixel spacing (for measuring branch cut length) (default: 1.0). */
    void ratioDxDy(const float);

    /** Get initial correlation threshold. */
    float initCorrThr() const;
    /** Set initial correlation threshold (default: 0.1). */
    void initCorrThr(const float);

    /** Get max correlation threshold. */
    float maxCorrThr() const;
    /** Set max correlation threshold (default: 0.9). */
    void maxCorrThr(const float);

    /** Get correlation threshold increment. */
    float corrThrInc() const;
    /** Set correlation threshold increment (default: 0.1). */
    void corrThrInc(const float);

    /** Get min connected component size as fraction of tile area. */
    float minCCAreaFrac() const;
    /** Set min connected component size as fraction of tile area (default: 0.003125). */
    void minCCAreaFrac(const float);

    /** Get number of bootstrap lines. */
    size_t numBsLines() const;
    /** Set number of bootstrap lines (default: 16). */
    void numBsLines(const size_t);

    /** Get bootstrapping min overlap area. */
    size_t minBsPts() const;
    /** Set bootstrapping min overlap area (default: 16). */
    void minBsPts(const size_t);

    /** Get bootstrap phase variance threshold. */
    float bsPhaseVarThr() const;
    /** Set bootstrap phase variance threshold (default: 8.0). */
    void bsPhaseVarThr(const float);

    /** 
     * \brief Unwrap the target interferogram.
     *
     * @param[out] unw Unwrapped phase
     * @param[out] ccl Connected component labels
     * @param[in] intf Interferogram
     * @param[in] corr Correlation
     * @param[in] seed Random state seed (default: 0)
     */
    void unwrap(
        isce::io::Raster & unw,
        isce::io::Raster & ccl,
        isce::io::Raster & intf,
        isce::io::Raster & corr,
        unsigned int seed = 0);

    // Compute residue charges.
    void getResidues(
        signed char * charge, 
        const float * phase, 
        const size_t length, 
        const size_t width);

    // Generate neutrons to guide the tree growing process.
    void genNeutrons(
        bool * neut, 
        const std::complex<float> * intf,
        const float * corr,
        const size_t length, 
        const size_t width);
    
    // Grow trees (make branch cuts).
    void growTrees(
        bool * tree,
        const signed char * charge,
        const bool * neut, 
        const size_t length,
        const size_t width,
        const unsigned int seed = 0);

    // Grow grass (find connected components and unwrap phase).
    template<bool DO_BOOTSTRAP>
    void growGrass(
        float * unw,
        uint8_t * ccl,
        bool * currcc,
        float * bsunw,
        uint8_t * bsccl, 
        LabelMap & labelmap,
        const float * phase, 
        const bool * tree, 
        const float * corr, 
        float corrthr,
        const size_t length,
        const size_t width);

private:
    // Configuration params
    size_t _NumBufLines = 3700;
    size_t _NumOverlapLines = 200;
    bool _UsePhaseGradNeut = false;
    bool _UseIntensityNeut = false;
    int _PhaseGradWinSize = 5;
    float _NeutPhaseGradThr = 3.f;
    float _NeutIntensityThr = 8.f;
    float _NeutCorrThr = 0.8f;
    int _NumTrees = 7;
    int _MaxBranchLen = 64;
    float _RatioDxDy = 1.f;
    float _InitCorrThr = 0.1f;
    float _MaxCorrThr = 0.9f;
    float _CorrThrInc = 0.1f;
    float _MinCCAreaFrac = 0.003125f;
    size_t _NumBsLines = 16;
    size_t _MinBsPts = 16;
    float _BsPhaseVarThr = 8.f;
};

} } }

// Get inline implementations.
#define ISCE_UNWRAP_ICU_ICU_ICC
#include "ICU.icc"
#undef ISCE_UNWRAP_ICU_ICU_ICC

#endif /* ISCE_UNWRAP_ICU_ICU_H */
