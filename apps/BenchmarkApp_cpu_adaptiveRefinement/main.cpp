//======================================================================================================================
//
//! \file main.cpp
//!
//! \brief Benchmark for adaptive (dynamic) mesh refinement of an actuator line simulation on CPUs.
//!
//! Turbine counterpart of waLBerla's AdaptiveMeshRefinementExample, stripped down for benchmarking in the spirit of
//! BenchmarkApp_gpu_staticRefinement: no flow statistics, no snapshots, no turbine output and no control. The only
//! (optional) output is VTK of the velocity field and of the domain decomposition. The flow is initialised uniformly;
//! the default case is a fully periodic box without any boundary cells.
//!
//! The initial block forest is created by the StaticRefinementHandler (turbine region on the finest level). Every
//! `refinementFrequency` time steps the block forest is refreshed:
//!   - blocks intersecting a static refinement region (turbine, AABBs, boundary layer) keep at least that level,
//!   - all other blocks are refined by one level where the velocity gradient measure exceeds `upperLimit` and
//!     coarsened by one level where it is below `lowerLimit`.
//! After a refresh that changed the forest, the non-uniform communication data, the flag field, the boundary index
//! vectors and the process-local turbine domain are rebuilt.
//!
//! Limitations: the depth of the forest is fixed (force scaling is relative to the finest level) and blocks are not
//! re-balanced (the wind farm only keeps turbines on the processes that owned turbine blocks at start-up).
//
//======================================================================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include <blockforest/Initialization.h>
#include <core/all.h>
#include <core/Environment.h>
#include <core/mpi/all.h>
#include <domain_decomposition/all.h>
#include <field/all.h>
#include <lbm/all.h>
#include <lbm_generated/field/AddToStorage.h>
#include <timeloop/all.h>
#include <vtk/all.h>

#include "conversion/Conversion.h"
#include "domain/BoundaryHandling.h"
#include "domain/BoundarySetup.h"
#include "domain/DomainInitialiser.h"
#include "domain/DomainSetup.h"
#include "farm_creator/WindFarm.h"
#include "farm_creator/ConfigTurbineCreator.h"
#include "output/Logging.h"
#include "walberla_helper/field/all.h"
#include "wind_turbine_core/ProjectDefines.h"

#include "waLBerlaWind_KernelInfo.h"

namespace turbine_core {

    static const uint_t fieldGhostLayers = 2;

    template<class Kernel_T>
    using KernelFieldInterpolator_T = projectors::KernelFieldInterpolator<field::Field<real_t>, Kernel_T>;

    template<class Kernel_T>
    using KernelDistributor_T = projectors::KernelFieldDistributor<field::Field<real_t>, Kernel_T>;

    using InterpolationKernel_T = projectors::SmoothedDiracDeltaKernel;
    using DistributionKernel_T = projectors::SmoothedDiracDeltaKernel;

    using WindFarm_T = WindFarm<topology::cpu::Tree, creator::ConfigTurbineCreator,
            ScalarField_T, VectorField_T, VectorField_T,
            KernelFieldInterpolator_T<InterpolationKernel_T>,
            KernelDistributor_T<DistributionKernel_T>>;

    using StaticRefinementHandler_T = refinement::StaticRefinementHandler<WindFarm_T>;

    //******************************************************************************************************************
    /*! Refinement criterion: static minimum level (static refinement regions) plus velocity gradient.
     *
     *  As in waLBerla's AdaptiveMeshRefinementExample, the measure of a cell is 0.25 * sum_d |u(x+e_d) - u(x-e_d)|^2,
     *  evaluated in fluid cells whose six direct neighbours are fluid as well. The target level is clamped to
     *  [currentLevel - 1, currentLevel + 1] and [requiredLevel, finestLevel], so the depth never changes.
     */
    //******************************************************************************************************************
    class TurbineAdaptiveRefinement {

    public:

        TurbineAdaptiveRefinement( StaticRefinementHandler_T * staticRefinementHandler,
                                   const BlockDataID & velocityFieldID, const BlockDataID & flagFieldID,
                                   const walberla::FlagUID & fluidFlagUID,
                                   const real_t lowerLimit, const real_t upperLimit,
                                   const uint_t boundaryLayerLevel, const uint_t finestLevel )
        : staticRefinementHandler_(staticRefinementHandler), velocityFieldID_(velocityFieldID), flagFieldID_(flagFieldID),
          fluidFlagUID_(fluidFlagUID), lowerLimitSq_(lowerLimit * lowerLimit), upperLimitSq_(upperLimit * upperLimit),
          boundaryLayerLevel_(boundaryLayerLevel), finestLevel_(finestLevel),
          gradientSamples_(std::make_shared<std::vector<real_t>>())
        {}

        //! Gradient measures of all blocks evaluated since the last clearGradientSamples() (shared between copies).
        const std::vector<real_t> & gradientSamples() const { return *gradientSamples_; }
        void clearGradientSamples() { gradientSamples_->clear(); }

        void operator()( std::vector< std::pair< const walberla::Block *, uint_t > > & minTargetLevels,
                         std::vector< const walberla::Block * > &, const walberla::BlockForest & forest ) {

            std::vector< StaticRefinementHandler_T::AABBPair > staticAABBs{};
            staticRefinementHandler_->calculateRefinementAABBs(staticAABBs);

            for( auto & [block, targetLevel] : minTargetLevels ) {

                const uint_t currentLevel = block->getLevel();

                uint_t requiredLevel{0};
                for( const auto & [aabb, level] : staticAABBs ) {
                    if( aabb.intersects(block->getAABB()) ) {
                        requiredLevel = std::max(requiredLevel, level);
                    }
                }
                if( boundaryLayerLevel_ > 0 && staticRefinementHandler_->checkIfBoundary(block, forest) ) {
                    requiredLevel = std::max(requiredLevel, boundaryLayerLevel_);
                }
                requiredLevel = std::min(requiredLevel, finestLevel_);

                uint_t level = currentLevel;
                if( currentLevel < requiredLevel ) {
                    level = currentLevel + uint_t(1);
                } else {
                    const real_t gradientSq = maxVelocityGradientSq(block);
                    gradientSamples_->push_back(std::sqrt(gradientSq));
                    // a non-finite measure (NaN) fails both comparisons and keeps the current level
                    if( gradientSq > upperLimitSq_ && currentLevel < finestLevel_ ) {
                        level = currentLevel + uint_t(1);
                    } else if( gradientSq < lowerLimitSq_ && currentLevel > requiredLevel ) {
                        level = currentLevel - uint_t(1);
                    }
                }

                targetLevel = level;
            }
        }

    private:

        real_t maxVelocityGradientSq( const walberla::Block * block ) const {

            const auto * u = block->getData< VectorField_T >( velocityFieldID_ );
            const auto * flags = block->getData< FlagField_T >( flagFieldID_ );
            WALBERLA_ASSERT_NOT_NULLPTR(u)
            WALBERLA_ASSERT_NOT_NULLPTR(flags)

            if( !flags->flagExists(fluidFlagUID_) )
                return real_t(0);
            const auto fluid = flags->getFlag(fluidFlagUID_);

            auto isFluid = [&](const cell_idx_t x, const cell_idx_t y, const cell_idx_t z) {
                return walberla::field::isFlagSet(flags->get(x, y, z), fluid);
            };

            const cell_idx_t xSize = walberla::cell_idx_c(u->xSize());
            const cell_idx_t ySize = walberla::cell_idx_c(u->ySize());
            const cell_idx_t zSize = walberla::cell_idx_c(u->zSize());

            real_t maxGradientSq{0};

            for( cell_idx_t z = cell_idx_t(1); z < zSize - cell_idx_t(1); ++z ) {
                for( cell_idx_t y = cell_idx_t(1); y < ySize - cell_idx_t(1); ++y ) {
                    for( cell_idx_t x = cell_idx_t(1); x < xSize - cell_idx_t(1); ++x ) {

                        if( !isFluid(x, y, z) ||
                            !isFluid(x + 1, y, z) || !isFluid(x - 1, y, z) ||
                            !isFluid(x, y + 1, z) || !isFluid(x, y - 1, z) ||
                            !isFluid(x, y, z + 1) || !isFluid(x, y, z - 1) )
                            continue;

                        for( uint_t d = 0; d < uint_t(3); ++d ) {
                            const real_t du_x = u->get(x + 1, y, z, d) - u->get(x - 1, y, z, d);
                            const real_t du_y = u->get(x, y + 1, z, d) - u->get(x, y - 1, z, d);
                            const real_t du_z = u->get(x, y, z + 1, d) - u->get(x, y, z - 1, d);
                            const real_t gradientSq = real_t(0.25) * (du_x * du_x + du_y * du_y + du_z * du_z);
                            if( !std::isfinite(gradientSq) ) {
                                return std::numeric_limits<real_t>::quiet_NaN();
                            }
                            maxGradientSq = std::max(maxGradientSq, gradientSq);
                        }
                    }
                }
            }

            return maxGradientSq;
        }

        StaticRefinementHandler_T * staticRefinementHandler_;

        walberla::ConstBlockDataID velocityFieldID_;
        walberla::ConstBlockDataID flagFieldID_;
        walberla::FlagUID fluidFlagUID_;

        real_t lowerLimitSq_;
        real_t upperLimitSq_;

        uint_t boundaryLayerLevel_;
        uint_t finestLevel_;

        std::shared_ptr<std::vector<real_t>> gradientSamples_;
    };


    int main(int argc, char** argv) {

        walberla::Environment walberlaEnv(argc, argv);

        walberla::logging::Logging::instance()->setLogLevel(walberla::logging::Logging::INFO);

        /// general simulation parameters
        auto globalConfig = walberlaEnv.config();
        auto parameters = globalConfig->getOneBlock("Parameters");

        uint_t timesteps = parameters.getParameter<uint_t>("timesteps", uint_t(10)); ++timesteps;
        const real_t remainingTimeLoggerFrequency = parameters.getParameter<real_t>("remainingTimeLoggerFrequency", real_t(3.0)); // in seconds

        const real_t length_SI = parameters.getParameter<real_t>("length_SI");
        const real_t length_LU = parameters.getParameter<real_t>("length_LU");
        const real_t velocity_SI = parameters.getParameter<real_t>("velocity_SI");
        const real_t velocity_LU = parameters.getParameter<real_t>("velocity_LU");
        const real_t viscosity_SI = parameters.getParameter<real_t>("viscosity_SI");
        const real_t density_SI = parameters.getParameter<real_t>("density_SI");

        Conversion::calculateConversionFactors(length_SI, length_LU, velocity_SI, velocity_LU, density_SI);
        Conversion::print();

        const real_t viscosity_LU = (viscosity_SI / density_SI) * Conversion::C_t() / Conversion::C_l() / Conversion::C_l();
        WALBERLA_CHECK_GREATER(viscosity_LU, real_t(0.0), "Negative LU viscosity - Did you calculate the conversion factors?")

        // omega on the coarsest level
        const real_t omega = walberla::lbm::collision_model::omegaFromViscosity(viscosity_LU);
        WALBERLA_LOG_INFO_ON_ROOT("omega = " << omega)

        /// ADAPTIVE REFINEMENT PARAMETERS

        auto domainConfig = globalConfig->getOneBlock("DomainSetup");
        uint_t finestLevel{0};
        uint_t boundaryLayerLevel{0};
        uint_t refinementFrequency{0};
        real_t lowerRefinementLimit{0};
        real_t upperRefinementLimit{0};
        if( domainConfig.getNumBlocks("Refinement") ) {
            auto refinementConfig = domainConfig.getOneBlock("Refinement");
            finestLevel = refinementConfig.getParameter<uint_t>("numLevels", uint_t(0));
            boundaryLayerLevel = refinementConfig.getParameter<uint_t>("BoundaryLayer", uint_t(0));
            if( refinementConfig.getNumBlocks("Adaptive") ) {
                auto adaptiveConfig = refinementConfig.getOneBlock("Adaptive");
                refinementFrequency = adaptiveConfig.getParameter<uint_t>("refinementFrequency", uint_t(100));
                lowerRefinementLimit = adaptiveConfig.getParameter<real_t>("lowerLimit");
                upperRefinementLimit = adaptiveConfig.getParameter<real_t>("upperLimit");
                WALBERLA_CHECK_LESS(lowerRefinementLimit, upperRefinementLimit,
                                    "Adaptive refinement: lowerLimit must be smaller than upperLimit.")
                // The measure is a velocity difference per cell, so it roughly halves when a block is refined.
                if( lowerRefinementLimit >= real_t(0.5) * upperRefinementLimit ) {
                    WALBERLA_LOG_WARNING_ON_ROOT("Adaptive refinement: lowerLimit >= upperLimit / 2. Since the gradient measure "
                                                 "halves under refinement, blocks may oscillate between levels.")
                }
            }
        }

        if( refinementFrequency > 0 && finestLevel == 0 ) {
            WALBERLA_LOG_WARNING_ON_ROOT("Adaptive refinement requested, but numLevels is 0. Adaptive refinement is disabled.")
            refinementFrequency = 0;
        }

        /// TURBINE

        auto windFarmConfig = globalConfig->getBlock("WindFarm");
        const std::string turbineOutputDirectory{};

        WindFarm_T farm(turbineOutputDirectory, windFarmConfig);
        farm.createGeometry();
        farm.calculateTurbineAABBs();

        /// BLOCKFOREST

        WALBERLA_LOG_INFO_ON_ROOT("Creating blockforest...")

        auto boundariesConfig = globalConfig->getOneBlock( "Boundaries" );
        domain::BoundarySetup boundarySetup(boundariesConfig);

        // The benchmark does not provide the atmospheric boundary layer parameters of wall functions and log-law inflow.
        const bool hasWalls = boundarySetup.environmentSetup() == EnvironmentSetup::Open ||
                              boundarySetup.environmentSetup() == EnvironmentSetup::Tunnel;
        WALBERLA_CHECK(!(hasWalls && boundarySetup.wallType() == WallSetup::WFB),
                       "Wall-function bounce is not supported in the benchmark app. Use wallType NoSlip.")
        WALBERLA_CHECK(boundarySetup.inflowType() != InflowSetup::InflowLogLaw,
                       "Log-law inflow is not supported in the benchmark app. Use UniformInflow or Periodic.")
        WALBERLA_CHECK(boundarySetup.inflowType() != InflowSetup::ShiftedPeriodic,
                       "Shifted periodicity is not supported in the benchmark app.")

        domain::DomainSetup domainSetup ( globalConfig, boundarySetup.periodicity() );
        WALBERLA_CHECK(!domainSetup.snapshotHandler_->loadSnapshot, "Snapshots are not supported in the benchmark app.")

        StaticRefinementHandler_T staticRefinementHandler{ domainConfig, &farm, boundarySetup.environmentSetup() };
        auto blocks = domainSetup.createStructuredBlockForest(staticRefinementHandler, fieldGhostLayers);

        WALBERLA_CHECK_EQUAL(blocks->getDepth(), finestLevel,
                             "The initial block forest must contain blocks on the finest level (numLevels).")

        farm.setBlockForest(blocks);
        farm.initialiseForceAndControlModel();

        /// FIELDS

        WALBERLA_LOG_INFO_ON_ROOT("Creating fields...")

        //layout - DO NOT CHANGE
        auto layout = codegen::KernelInfo::layout;
        const StorageSpecification_T storageSpec{};

        auto allocator = std::make_shared< walberla::field::AllocateAligned<real_t, 64> >();

        BlockDataID densityFieldID = walberla::field::addToStorage<ScalarField_T>(blocks, "density field", real_t(1), layout, fieldGhostLayers, allocator);
        BlockDataID flagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field", fieldGhostLayers);
        BlockDataID forceFieldID = walberla::field::addToStorage<VectorField_T>(blocks, "force field", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID pdfFieldID = walberla::lbm_generated::addPdfFieldToStorage(blocks, "pdf field", storageSpec, fieldGhostLayers, layout, walberla::Set<walberla::SUID>::emptySet(), walberla::Set<walberla::SUID>::emptySet(), allocator);
        BlockDataID velocityFieldID = walberla::field::addToStorage<VectorField_T>(blocks, "velocity field", real_t(0), layout, fieldGhostLayers, allocator);

        /// DOMAIN INITIALISATION (uniform flow)

        WALBERLA_LOG_INFO_ON_ROOT("Initialise domain...")
        {
            const Vector3<real_t> initialVelocity = globalConfig->getOneBlock("Initialisation").getParameter<Vector3<real_t>>("initialVelocity");
            domain::UniformInitialisation initialiser(blocks, initialVelocity);
            using PdfSetter_T = walberla::pystencils::waLBerlaWind_PdfSetter;
            PdfSetter_T setter(densityFieldID, forceFieldID, pdfFieldID, velocityFieldID);
            initialiser.setViaVelocityField<VectorField_T, PdfSetter_T>(velocityFieldID, setter);
        }

        SweepCollection_T sweepCollection( blocks, densityFieldID, forceFieldID, pdfFieldID, velocityFieldID, omega );

        /// BOUNDARIES

        WALBERLA_LOG_INFO_ON_ROOT("Setting up boundaries...")

        // Has to be redone after every refresh, since new blocks start with a migrated (coarsened/refined) flag field.
        auto setupFlagField = [&]() {
            for( auto & block : *blocks ) {
                auto * flagField = block.getData<FlagField_T>(flagFieldID);
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(flagField, flagField->get(x, y, z) = flag_t(0);)
            }
            boundarySetup.fillFlagFieldFromConfig<FlagField_T>(blocks, flagFieldID, FluidFlagUID,
                                                               NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID,
                                                               LogLawInflowFlagUID, OutflowFlagUID);
        };
        setupFlagField();

        // Wall-function and log-law parameters are unused (checked above); dummy values satisfy the generated constructor.
        const auto & inflowVelocity = boundarySetup.inflowVelocity();
        const real_t dummyRoughnessLength{1e-3};
        auto velocityInit = boundary::velocityInit(dummyRoughnessLength, real_t(0.42), real_t(0));
        BoundaryCollection_T boundaryCollection(
        	blocks, flagFieldID, pdfFieldID, FluidFlagUID, velocityFieldID,
        	uint32_t(1), real_t(1e-3), dummyRoughnessLength,
        	inflowVelocity[0], inflowVelocity[1], inflowVelocity[2], velocityInit
        );

        auto refillBoundaries = [&]() {
            boundaryCollection.waLBerlaWind_NoSlipObject->fillFromFlagField<FlagField_T>(blocks, flagFieldID, NoSlipFlagUID, FluidFlagUID);
            boundaryCollection.waLBerlaWind_FreeSlipObject->fillFromFlagField<FlagField_T>(blocks, flagFieldID, SymmetryFlagUID, FluidFlagUID);
            boundaryCollection.waLBerlaWind_WFBObject->fillFromFlagField<FlagField_T>(blocks, flagFieldID, WFBFlagUID, FluidFlagUID);
            boundaryCollection.waLBerlaWind_UniformUBBObject->fillFromFlagField<FlagField_T>(blocks, flagFieldID, UniformInflowFlagUID, FluidFlagUID);
            boundaryCollection.waLBerlaWind_LogLawUBBObject->fillFromFlagField<FlagField_T>(blocks, flagFieldID, LogLawInflowFlagUID, FluidFlagUID);
            boundaryCollection.waLBerlaWind_OutflowObject->fillFromFlagField<FlagField_T>(blocks, flagFieldID, OutflowFlagUID, FluidFlagUID);
        };

        farm.setFieldIDs(densityFieldID, velocityFieldID, forceFieldID);

        /// COMMUNICATION

        WALBERLA_LOG_INFO_ON_ROOT("Set up communication...")

        // The non-uniform buffered scheme detects block forest refreshes itself (modification stamp).
        auto communication = std::make_shared<walberla::blockforest::communication::NonUniformBufferedScheme<CommunicationStencil_T>>(blocks);
        auto pdfPackInfo = walberla::lbm_generated::setupNonuniformPdfCommunication<PdfField_T>(blocks, pdfFieldID);
        communication->addPackInfo( pdfPackInfo );
        communication->addPackInfo( std::make_shared<walberla::field::refinement::PackInfo<ScalarField_T, CommunicationStencil_T>>(densityFieldID) );
        communication->addPackInfo( std::make_shared<walberla::field::refinement::PackInfo<VectorField_T, CommunicationStencil_T>>(velocityFieldID) );

        /// TIMELOOP

        WALBERLA_LOG_INFO_ON_ROOT("Creating time loop...")

        using TimingPolicy_T = walberla::timing::WcPolicy;
        auto timeloop = walberla::timeloop::SweepTimeloop<TimingPolicy_T>(blocks->getBlockStorage(), timesteps);

        farm.setTimeloop(&timeloop);

        // The flow driver also resets the force field before the turbine forces are spread (driverType None: zero force).
        walberla::wind::FlowDriverCollection flowDriver(blocks, &timeloop, globalConfig, domainSetup, forceFieldID, velocityFieldID, fieldGhostLayers);
        for(auto & block : *blocks) { flowDriver(&block); }

        uint_t maxKernelWidth = 1;
        if constexpr (std::is_same_v<WindFarm_T::Interpolator, KernelFieldInterpolator_T<InterpolationKernel_T>>) {
            maxKernelWidth = std::max({maxKernelWidth, InterpolationKernel_T::xWidth, InterpolationKernel_T::yWidth, InterpolationKernel_T::zWidth});
        }
        if constexpr (std::is_same_v<WindFarm_T::Distributor, KernelDistributor_T<DistributionKernel_T >>) {
            maxKernelWidth = std::max({maxKernelWidth, DistributionKernel_T::xWidth, DistributionKernel_T::yWidth, DistributionKernel_T::zWidth});
        }

        walberla::Config::Blocks turbineBlocks{};
        windFarmConfig.getBlocks(turbineBlocks);
        uint_t diameter = 0;
        for (auto & turbineConfig : turbineBlocks) {
            if(std::string(turbineConfig.getParameter<std::string>( "aeroModel" )) == "rotorDisk") {
                diameter = math::max(diameter, turbineConfig.getParameter<uint_t>("diameter_LU"));
            }
        }

        auto forBlocksOnLevel = [&](const uint_t level, const std::function<void(walberla::IBlock *)> & func) {
            std::vector<walberla::Block *> blocksOnLevel;
            blocks->getBlocks(blocksOnLevel, level);
            for (auto block : blocksOnLevel) {
                func(block);
            }
        };

        walberla::lbm_generated::BasicRecursiveTimeStep<PdfField_T, SweepCollection_T, BoundaryCollection_T> recursiveTimestep (
        	blocks, pdfFieldID, sweepCollection, boundaryCollection, communication, pdfPackInfo );

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::UPDATE_DISCRETISATION); }, "Update discretisation");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
            forBlocksOnLevel(level, [&](walberla::IBlock * block) { farm.evaluateDensityAndVelocity(block); });
        }, "Evaluate density and velocity");
        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.syncNextNeighbour(mpi::SYNC_MACRO, 1, false); }, "Sync turbine neighbours");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
            forBlocksOnLevel(level, [&](walberla::IBlock * block) { flowDriver(block); });
        }, "Flow driver");
        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::CALCULATE_FORCES); }, "Calculate forces");
        recursiveTimestep.addPostCollisionFunction([&](uint_t) {
            farm.syncNextNeighbour(mpi::SYNC_FORCE, maxKernelWidth + uint_t(real_t(diameter) / real_t(2.0)), false);
        }, "Sync forces");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
            forBlocksOnLevel(level, [&](walberla::IBlock * block) { farm.spreadForces(block); });
        }, "Spread forces");
        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::RESET_COMMUNICATION_DATA); }, "Reset communication data");

        /// ADAPTIVE MESH REFINEMENT

        auto & forest = blocks->getBlockForest();

        forest.recalculateBlockLevelsInRefresh(true);
        forest.allowMultipleRefreshCycles(true);
        forest.reevaluateMinTargetLevelsAfterForcedRefinement(true);
        forest.checkForEarlyOutInRefresh(true);
        forest.checkForLateOutInRefresh(true);
        forest.allowRefreshChangingDepth(false);  // force scaling is relative to the finest level
        forest.alwaysRebalanceInRefresh(false);   // turbines only exist on the processes that owned turbine blocks at start-up

        TurbineAdaptiveRefinement refinementCriterion( &staticRefinementHandler, velocityFieldID, flagFieldID, FluidFlagUID,
                                                       lowerRefinementLimit, upperRefinementLimit,
                                                       boundaryLayerLevel, finestLevel );
        forest.setRefreshMinTargetLevelDeterminationFunction(refinementCriterion);

        // number of blocks and cells per level
        auto blocksPerLevel = [&]() {
            std::vector<uint_t> numBlocks(finestLevel + 1, uint_t(0));
            for( uint_t level = 0; level <= finestLevel; ++level ) {
                numBlocks[level] = forest.getNumberOfBlocks(level);
            }
            walberla::mpi::allReduceInplace(numBlocks, walberla::mpi::SUM);
            return numBlocks;
        };
        const uint_t cellsPerBlock = blocks->getNumberOfXCellsPerBlock() * blocks->getNumberOfYCellsPerBlock() * blocks->getNumberOfZCellsPerBlock();

        auto logBlockStructure = [&]() {
            const auto numBlocks = blocksPerLevel();
            std::stringstream ss;
            for( uint_t level = 0; level <= finestLevel; ++level ) {
                ss << " | L" << level << ": " << numBlocks[level];
            }
            WALBERLA_LOG_INFO_ON_ROOT("Blocks per level at time step " << timeloop.getCurrentTimeStep() << ss.str())
        };

        auto logGradientStatistics = [&]() {
            auto samples = walberla::mpi::gatherv(refinementCriterion.gradientSamples(), 0);
            WALBERLA_ROOT_SECTION() {
                if( samples.empty() ) return;
                std::sort(samples.begin(), samples.end());
                const auto nAbove = std::count_if(samples.begin(), samples.end(), [&](real_t g) { return g > upperRefinementLimit; });
                const auto nBelow = std::count_if(samples.begin(), samples.end(), [&](real_t g) { return g < lowerRefinementLimit; });
                WALBERLA_LOG_INFO("Gradient measure: min " << samples.front()
                                  << ", median " << samples[samples.size() / 2] << ", max " << samples.back()
                                  << " | " << nAbove << " above upperLimit, " << nBelow << " below lowerLimit")
            }
        };

        walberla::WcTimingPool refinementTiming;
        uint_t numRefreshes{0};
        // cell updates over the whole run (fine blocks perform 2^level updates per coarse time step)
        double cellUpdates{0};

        auto adaptiveRefinementStep = [&]() {

            const uint_t stampBefore = forest.getModificationStamp();

            refinementCriterion.clearGradientSamples();
            refinementTiming["refresh"].start();
            forest.refresh();
            refinementTiming["refresh"].end();
            logGradientStatistics();

            if( forest.getModificationStamp() == stampBefore ) {
                return;
            }
            ++numRefreshes;

            refinementTiming["re-initialisation"].start();
            for( auto & block : *blocks ) {
                pdfPackInfo->recalculateNonuniformCommData(&block);
            }
            setupFlagField();
            refillBoundaries();
            farm.setBlockForest(blocks);
            refinementTiming["re-initialisation"].end();

            WALBERLA_CHECK_EQUAL(blocks->getDepth(), finestLevel, "Adaptive refinement must not change the depth of the block forest.")

            logBlockStructure();
        };

        logBlockStructure();

        if( refinementFrequency > 0 ) {
            WALBERLA_LOG_INFO_ON_ROOT("Adaptive mesh refinement every " << refinementFrequency << " time steps "
                                      << "(lowerLimit = " << lowerRefinementLimit << ", upperLimit = " << upperRefinementLimit << ")")
            timeloop.addFuncBeforeTimeStep([&, refinementFrequency]() {
                const uint_t step = timeloop.getCurrentTimeStep();
                if( step > uint_t(0) && step % refinementFrequency == 0 ) {
                    adaptiveRefinementStep();
                }
            }, "Adaptive mesh refinement");
        }

        // process-local count, reduced after the time loop
        timeloop.addFuncBeforeTimeStep([&]() {
            for( uint_t level = 0; level <= finestLevel; ++level ) {
                cellUpdates += double(forest.getNumberOfBlocks(level)) * double(cellsPerBlock) * double(uint_t(1) << level);
            }
        }, "Count cell updates");

        // The recursive time step queries the blocks per level in every time step and picks up a changed forest.
        timeloop.addFuncBeforeTimeStep(recursiveTimestep, "Recursive LBM time step");

        timeloop.addFuncAfterTimeStep( walberla::makeSharedFunctor( walberla::field::makeStabilityChecker< PdfField_T, FlagField_T >(globalConfig, blocks, pdfFieldID, flagFieldID, FluidFlagUID )), "LBM stability check" );
        timeloop.addFuncAfterTimeStep( walberla::timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "Remaining time logger" );

        /// VTK OUTPUT (velocity field and domain decomposition)

        // Both outputs iterate over the current blocks at every write, so they follow the adaptive block structure.
        // The velocity is the one computed in the last stream-collide sweep. Set writeFrequency 0 for benchmarking.
        uint_t vtkWriteFrequency{0};
        std::string vtkBaseFolder{"vtk_out"};
        if( globalConfig->getNumBlocks("Output") && globalConfig->getOneBlock("Output").getNumBlocks("VTK") ) {
            auto vtkConfig = globalConfig->getOneBlock("Output").getOneBlock("VTK");
            vtkWriteFrequency = vtkConfig.getParameter<uint_t>("writeFrequency", uint_t(0));
            vtkBaseFolder = std::string(vtkConfig.getParameter<std::string>("baseFolder", vtkBaseFolder));
        }

        if( vtkWriteFrequency > 0 ) {
            WALBERLA_LOG_INFO_ON_ROOT("VTK output of velocity and domain decomposition every " << vtkWriteFrequency
                                      << " time steps to '" << vtkBaseFolder << "'")

            // unstructured output (forcePVTU) to support blocks on different levels, as in AdaptiveMeshRefinementExample
            auto velocityOutput = walberla::vtk::createVTKOutput_BlockData(*blocks, "velocity", vtkWriteFrequency, 0, true,
                                                                           vtkBaseFolder, "simulation_step", false, true, true, false, 0);
            velocityOutput->addCellDataWriter(std::make_shared<walberla::field::VTKWriter<VectorField_T, float>>(velocityFieldID, "velocity"));
            walberla::field::FlagFieldCellFilter<FlagField_T> fluidFilter(flagFieldID);
            fluidFilter.addFlag(FluidFlagUID);
            velocityOutput->addCellInclusionFilter(fluidFilter);
            timeloop.addFuncAfterTimeStep(walberla::vtk::writeFiles(velocityOutput), "VTK velocity output");

            auto domainDecompositionOutput = walberla::vtk::createVTKOutput_DomainDecomposition(
                    *blocks, "domain_decomposition", vtkWriteFrequency, vtkBaseFolder, "simulation_step", false, true, true, false);
            timeloop.addFuncAfterTimeStep(walberla::vtk::writeFiles(domainDecompositionOutput), "VTK domain decomposition output");
        }

        WALBERLA_MPI_WORLD_BARRIER()
        WALBERLA_LOG_INFO_ON_ROOT("Running timeloop...")

        walberla::timing::TimingPool<TimingPolicy_T> timing;

        walberla::WcTimer timer;
        timer.start();
        for( uint_t i = 0; i < timesteps; ++i ) {
            timeloop.singleStep(timing);
        }
        timer.end();

        double time = timer.max();
        walberla::mpi::reduceInplace(time, walberla::mpi::MAX);
        walberla::mpi::reduceInplace(cellUpdates, walberla::mpi::SUM);

        const auto timeloopTiming = timing.getReduced();
        WALBERLA_LOG_INFO_ON_ROOT("Timeloop timing:\n" << *timeloopTiming)

        if( refinementFrequency > 0 ) {
            const auto reducedRefinementTiming = refinementTiming.getReduced();
            WALBERLA_LOG_INFO_ON_ROOT("Adaptive refinement timing (" << numRefreshes << " refreshes changed the forest):\n" << *reducedRefinementTiming)
        }

        // Counts the cells of the final forest only, see the MLUPS below for the adaptive forest.
        walberla::lbm::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldID, FluidFlagUID);
        performance.logResultOnRoot(timesteps, time);

        WALBERLA_LOG_INFO_ON_ROOT("Adaptive forest: " << cellUpdates * 1e-6 / time << " MLUPS (cell updates of the actual forest in every time step, "
                                  << "including time spent in refreshes)")

        logging::WindTurbineApplicationLogger<domain::DomainSetup, domain::BoundarySetup, codegen::KernelInfo, TimingPolicy_T> logger {
            domainSetup, boundarySetup, windFarmConfig, &timing, performance.loggingString(timesteps, time)
        };

        return EXIT_SUCCESS;

    } // main

} // namespace turbine_core

int main(int argc, char** argv) {
    return turbine_core::main(argc, argv);
}
