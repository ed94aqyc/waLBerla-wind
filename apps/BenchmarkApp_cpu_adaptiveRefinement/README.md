# Adaptive mesh refinement benchmark (CPU)

Turbine counterpart of waLBerla's `apps/examples/AdaptiveMeshRefinementExample`, stripped down for benchmarking like
`BenchmarkApp_gpu_staticRefinement`. It has no turbine output, no flow statistics, no snapshots, no control and no
atmospheric boundary layer. The only optional output is VTK of the velocity field and the domain decomposition. The
flow starts uniform. The default `input.prm` is a fully periodic box without boundary cells, containing one actuator
line turbine (MEXICO rotor) and one refinement level. It runs for a few minutes on 4 cores.

The GPU version is [`BenchmarkApp_gpu_adaptiveRefinement`](../BenchmarkApp_gpu_adaptiveRefinement/README.md).

## How it works

* **Initial block structure.** It comes from the `StaticRefinementHandler`, with the turbine region on the finest
  level (`numLevels`), plus optional `AABB`s and `BoundaryLayer`.
* **Refresh.** Every `refinementFrequency` time steps, `blockforest->refresh()` runs with the criterion
  `TurbineAdaptiveRefinement`:
  * Blocks that intersect a static refinement region keep at least that region's level, so the turbine region stays
    on the finest level.
  * All other blocks are refined by one level where the velocity gradient measure exceeds `upperLimit`, and coarsened
    by one level where it is below `lowerLimit`. The measure is the one from the waLBerla example: the central
    velocity difference per cell, in LU.
  * Choose `lowerLimit < upperLimit / 2`. The measure roughly halves under refinement, so closer limits let blocks
    oscillate between levels. waLBerla only coarsens complete octets of sibling blocks.
* **After a refresh that changed the forest,** the app rebuilds the following. The recursive time step and the
  communication scheme adapt automatically.
  * the non-uniform PDF communication data (`recalculateNonuniformCommData`)
  * the flag field and the boundary index vectors
  * the process-local turbine domain (`farm.setBlockForest`)

## Output

* **VTK** (`Output { VTK { writeFrequency; baseFolder; } }`, `writeFrequency 0` disables it): every
  `writeFrequency` time steps, the app writes `<baseFolder>/velocity.pvd` and `<baseFolder>/domain_decomposition.pvd`.
  * The velocity is written as unstructured grid (PVTU), one piece per block, so blocks on different levels are
    shown with their own resolution.
  * The domain decomposition has the cell data `Level` and `Process`. It shows how the forest adapts and how blocks
    are distributed over the processes.
  * Both follow the current block structure, so every output reflects the forest after the latest refresh.
  * With the default case and `writeFrequency 250`, this writes about 160 MB and costs about 1 % of the run time.
    Set it to 0 for performance measurements.
* At every refresh: blocks per level and the distribution of the gradient measure.
* At the end:
  * the timing pools of the time loop and of the refreshes;
  * waLBerla's `PerformanceEvaluation`, which counts the cells of the *final* forest only;
  * an "Adaptive forest" MLUPS value. It counts the cells of the actual forest in every time step, weighted by
    `2^level`, and includes the time spent in refreshes.

## Limitations

* **Fixed depth** (`allowRefreshChangingDepth(false)`). Force scaling is relative to the finest level.
* **No load balancing** (`alwaysRebalanceInRefresh(false)`). `ConfigTurbineCreator` only keeps turbines on processes
  that owned turbine blocks at start-up.
* **Unsupported boundaries.** Wall-function bounce and log-law inflow are not supported, because their atmospheric
  boundary layer parameters were removed. Shifted periodicity is not supported either. No-slip, free-slip, uniform
  inflow and outflow work.
* **Data transfer on refresh.** PDFs of refined or coarsened blocks are copied or averaged without rescaling, as in
  the upstream example.

## Requirements

* A waLBerla with AMR support in `lbm_generated`, which means upstream commits `ed7f10eb6` and `fecd72661` (waLBerla
  3.7 / master has both). `apps/CMakeLists.txt` skips the app otherwise.
* pystencils/lbmpy 1.4.x for the legacy code generation.
