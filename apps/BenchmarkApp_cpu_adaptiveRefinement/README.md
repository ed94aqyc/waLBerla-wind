# Adaptive mesh refinement benchmark (CPU)

Turbine counterpart of waLBerla's `apps/examples/AdaptiveMeshRefinementExample`, stripped down for benchmarking like
`BenchmarkApp_gpu_staticRefinement`. It has no turbine output, no flow statistics, no snapshots, no control and no
atmospheric boundary layer. The only optional output is VTK of the velocity field and the domain decomposition. The
flow starts uniform. The case is a fully periodic box without boundary cells, containing one actuator line turbine
(MEXICO rotor, D = 15 cells) and one refinement level.

* **Domain.** 256 x 120 x 120 cells of the finest level, which is 8 x 3 x 3 root blocks of 32 x 40 x 40 cells
  (`cellsPerBlock <16,20,20>`). The turbine sits in the middle root block in y and z, with at least 11.8 cells to
  the root block faces.
* **Refinement.** Initially, only the turbine's root block is refined, giving 71 + 8 blocks. The wake then refines
  one more root block of the middle row (in y and z) about every 300 time steps, reaching 65 + 56 blocks after
  2000 steps.
  * The blocks are wider in y and z than in x, so the expanding wake stays within the middle row and only those
    blocks refine.
  * With 32-cell root blocks in y and z, the wake spilled into neighbouring rows after about 1700 steps.
* **Run time.** The 2001 time steps take about 8 minutes on 4 cores. One flow-through of the domain takes about 4400
  time steps.

## Cases

Three parameter files compare uniform, static and adaptive refinement:

* They share the domain, the block size, the turbine and all parameters (2001 time steps, VTK output every 100
  steps). Only the `Refinement` block differs.
* All three use `numLevels 1`, so a time step, `omega`, the turbine forces and the cell-update count mean the same
  in every case. In the uniform case, level 0 is simply empty.

| File | Refinement | Blocks (L0 + L1) | Cells (L0 + L1) | Cell updates per time step | Time, 4 cores | Time, 1 GPU |
|---|---|---|---|---|---|---|
| `adaptive.prm` | at run time, where the wake develops | 71 + 8 → 65 + 56 | 505,600 → 774,400 (13.7 → 21.0 %) | 556,800 → 1,132,800, about 0.83 M on average | 7.6 min | 4.9 min |
| `static.prm` | middle row of root blocks (`AABB { global { ... } }`) | 64 + 64 | 409,600 + 409,600 = 819,200 (22.2 %) | 1,228,800 | about 8 min | about 6 min |
| `uniform.prm` | whole domain | 0 + 576 | 3,686,400 (100 %) | 7,372,800 | about 35 min | about 25 min |

* Each block has 16 x 20 x 20 = 6400 cells. A block on level 1 is updated twice per time step.
* The percentages are relative to the 3,686,400 cells of the domain at the finest level.
* The adaptive average is estimated from the block history. A full run prints the exact value.

* The times are for 2001 time steps.
  * Adaptive: measured with VTK output every 100 steps.
  * Static and uniform: extrapolated from 200 and 50/100 steps without VTK output.
* VTK output every 100 steps adds about 300 MB per output in the uniform case (about 6 GB per run), so set
  `-Output.VTK.writeFrequency=0` for performance measurements.
* The static case refines the same region that the adaptive case reaches after about 2000 steps (one more root
  block in x). Its refinement box is one cell smaller than the root blocks of the middle row, so that the
  neighbouring rows are not touched.

```
mpirun -np 4 ./BenchmarkApp_cpu_adaptive_refinement adaptive.prm   # or static.prm, uniform.prm, *_large.prm
```

The GPU version is [`BenchmarkApp_gpu_adaptiveRefinement`](../BenchmarkApp_gpu_adaptiveRefinement/README.md).

### Large cases

`adaptive_large.prm`, `static_large.prm` and `uniform_large.prm` are the same three cases at 3.75 times the
resolution in every direction, with the same root blocks and the same physical time. They are identical to the GPU
files of the same name, except for `gpuEnabledMPI` and the stability check frequency. The
[GPU README](../BenchmarkApp_gpu_adaptiveRefinement/README.md#large-cases-for-modern-gpus) lists the scaled
parameters.

| File | Cells | Cell updates per time step | Memory | Time, 4 cores of this machine |
|---|---|---|---|---|
| `uniform_large.prm` | 199.6 M | 399.2 M | about 59 GB | about 140 h |
| `static_large.prm` | 44.4 M | 66.5 M | about 13 GB (measured) | about 23 h |
| `adaptive_large.prm` | 27.4 M at the start, growing with the wake | 30.2 M at the start | 8.1 GB at the start | |

* The app needs about 300 bytes per cell.
* The times are extrapolated from 6 MLUPS (3 time steps of the static case on 4 cores) to 7501 time steps. Run
  these cases on a cluster.
* The forest is not rebalanced after a refresh, so new fine blocks stay on the process of their parent. Use at most
  one process per block: 79 blocks at the start of the adaptive case, 128 in the static case and 576 in the uniform
  case.

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
  * With the default case and `writeFrequency 100`, this writes about 1.1 GB and costs a few percent of the run
    time.
    Set it to 0 for performance measurements.
* At time step 0 and after every refresh that changed the forest: blocks and cells per level, the total number of
  cells (also as a fraction of a uniform finest-level domain), and the cell updates per time step.
* At every refresh: the distribution of the gradient measure.
* At the end:
  * the timing pools of the time loop and of the refreshes;
  * waLBerla's `PerformanceEvaluation`, which counts the cells of the *final* forest only;
  * the total and average number of cell updates per time step;
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
