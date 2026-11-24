# Adaptive mesh refinement benchmark (GPU)

GPU version of [`BenchmarkApp_cpu_adaptiveRefinement`](../BenchmarkApp_cpu_adaptiveRefinement/README.md). It runs the
same cases (`adaptive.prm`, `static.prm`, `uniform.prm`) with the same refinement criterion, the same parameters and
the same output. See the CPU README for how the adaptive refinement works, for the cases and for the output.

The kernels are those of `BenchmarkApp_gpu_staticRefinement`: cumulant D3Q27 with QR subgrid model and Guo forcing,
fast divisions and square roots, and a compile-time GPU block size of (128, 1, 1). One difference: the streaming
pattern is pull, not AA (see below).

## Running

Use one MPI process per GPU:

```
mpirun -np 1 ./BenchmarkApp_gpu_adaptive_refinement adaptive.prm   # or static.prm, uniform.prm, *_large.prm
```

**Several GPUs per node:** give every process only its own GPU:

```
mpirun -np 2 sh -c 'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK exec ./BenchmarkApp_gpu_adaptive_refinement adaptive.prm'
```

* With the GTX 980 Ti (Maxwell) on this machine, a process whose current device is not device 0 computes a NaN
  turbine AABB in `GPUTurbineTopology::getAABB()`. That process then builds an unrefined forest, and the app stops
  with "The initial block forest must contain blocks on the finest level".
* The cause is probably the `__managed__` globals of the GPU turbine topology on pre-Pascal GPUs. It is not specific
  to this app and is not investigated further.
* With the command above, waLBerla prints a harmless warning ("Too many processes started per node").

## Large cases for modern GPUs

`adaptive_large.prm`, `static_large.prm` and `uniform_large.prm` are the same three cases at 3 times the resolution
in every direction. They keep the same 8 x 3 x 3 root blocks, the same block structure and the same physical time.

| Case | Cells | Cell updates per time step | GPU memory | Pinned host memory |
|---|---|---|---|---|
| `uniform_large.prm` | 99.5 M | 199.1 M | about 39 GB | about 17 GB |
| `static_large.prm` | 22.1 M | 33.2 M | about 8.7 GB | about 3.7 GB |
| `adaptive_large.prm` | 13.7 M at the start, growing with the wake | 15.0 M at the start | 5.5 GB at the start (measured) | 2.3 GB at the start |

* **Scaled parameters.**
  * Grid: `cellsPerBlock <48,60,60>` (172,800 cells, 27 times the small case), `domainSize <768,360,360>`.
  * Turbine: `basePoint <144,180,180>`, `diameter_LU` and `length_LU` of 45 (D = 45 cells on the finest level).
  * Time: time steps, refinement frequency and stability check frequency x 3 (6001 time steps, refinement every 300
    steps).
  * Refinement limits / 3, because the velocity difference per cell scales with the cell size.
  * SI parameters and LU velocity are unchanged.
* **Wake.**
  * The wake reaches the end of the domain (x = 768, 624 cells behind the rotor) after at least 10,800 time steps
    (6.2 s) at the free-stream velocity. Probably closer to 15,000 steps, since the wake front is slower.
  * The 6001 time steps bring it to about the middle of the domain. Set `-Parameters.timesteps=...` for longer runs.
  * The domain is periodic, so the wake re-enters at x = 0 and reaches the turbine again after one flow-through
    (13,300 steps at the free-stream velocity).
* **Memory.**
  * About 68 MB per block on the GPU (single precision), measured with the start of the adaptive case on a GTX 980
    Ti, plus 29 MB per block of pinned host memory for the CPU mirrors.
  * A field row of 48 + 4 ghost cells is padded to 64 floats (256 bytes, the pitch alignment of waLBerla's GPU
    fields). That is 23 % more GPU memory than unpadded. 60 cells in x would avoid it, but y and z would then have
    to be 75, and the cells per block must be even.
  * The uniform case is too tight for a 40 GB GPU. Use a larger GPU, or two GPUs with one MPI process each.
* **Output.** VTK output is off (`writeFrequency 0`); one output of the uniform case would be about 8 GB. Every
  stability check (every 1200 steps) copies all PDFs to the host, up to 13 GB.
* **Build.** Set `CMAKE_CUDA_ARCHITECTURES` for the target GPU, for example 80 for A100 or 90 for H100, and use single
  precision (`WALBERLA_DOUBLE_ACCURACY=OFF`).
* **Status.**
  * The start of the adaptive case ran for 3 time steps on one GTX 980 Ti (refresh in every step, 31.6 MLUPS).
  * The static and adaptive cases ran for 3 time steps with the CPU app. The forests and cell counts are as in the
    table.
  * Static and uniform don't fit on the 6 GB GPUs of this machine and have not run on a GPU.

## Differences to the CPU version

* **CPU mirrors.** The simulation runs on the GPU fields. Every field has a CPU mirror in pinned memory. The mirrors
  are used:
  * to migrate data during `blockforest->refresh()`, because the GPU fields of new blocks are re-allocated, not
    migrated;
  * as input of the refinement criterion and of the VTK output;
  * by the stability checker.
* **Refresh.** Before every refresh, PDFs, density and velocity are copied to the CPU. After a refresh that changed
  the forest, they are copied back, and the app rebuilds the following:
  * the GPU communication data (`recalculateNonuniformGPUCommData`)
  * the flag field and the boundary index vectors
  * the process-local turbine domain (`farm.setBlockForest`) and the turbines' cached GPU communication setup
    (`farm.resetTurbineCommunication`)
  * the per-level resources of the recursive time step (`update()`)
* **Pull instead of AA streaming.** For in-place patterns on GPUs, `lbm_generated` generates "block-wise" kernels.
  * They cache device pointers of all blocks at set-up: `NonuniformGeneratedGPUPdfPackInfo` for the local
    equal-level communication and, after `initialiseBlockPointer()`, the sweep collection.
  * These caches are never rebuilt after a refresh, so they would point to freed memory. A `static_assert` in
    `main.cu` rejects block-wise kernels.
* **Stability checker and VTK.** The PDFs are copied to the CPU only at the steps where the stability checker
  checks. The velocity is copied only at the steps where VTK output is written.
* **Timing.** The time loop uses the `DeviceSynchronizePolicy` and the per-kernel functions of
  `BasicRecursiveTimeStepGPU::addRefinementToTimeLoop`, as in `BenchmarkApp_gpu_staticRefinement`. The adaptive
  refinement timing additionally lists the GPU-CPU transfers.
* **Parameter `gpuEnabledMPI`** (in `Parameters`, default `false`): enables CUDA-aware MPI for the PDF communication.

## Validation and performance

The default case (256 x 120 x 120, 2001 time steps) takes 4.9 minutes on one GTX 980 Ti, compared with about
8 minutes on 4 CPU cores.

* Throughput is 5.7 MLUPS. The VTK output every 100 steps takes 12 % of the time.
* The forest is identical to the CPU app at every refresh.

The following was measured with the previous default case (256 x 64 x 64, turbine at <48, 32, 32>), with
`refinementFrequency 50`, 1000 time steps, GTX 980 Ti GPUs and single precision:

* **Forest.** The forest evolves exactly as in the CPU app (refreshes at steps 200, 550, 700 and 900; 60 to 130
  blocks). This holds on 1 and on 2 GPUs.
* **Velocity.** Compared with the CPU app on the same cells, the maximum difference is 1e-4 of the maximum wake
  deficit after 300 steps without refinement. With refinement, it grows smoothly to 1.3e-3 after 1000 steps, with no
  jumps at the refreshes.
* **Performance.** The default case is far too small for a GPU: 3.8 MLUPS on 1 GPU and 4.9 MLUPS on 2 GPUs. The time
  goes to communication and turbine kernels on the 16³ blocks.
  * With `cellsPerBlock <32,32,32>` and `domainSize <512,128,128>`, 1 GPU reaches 35 MLUPS, and the equal-level
    communication still takes 40 % of the time.
  * Use larger blocks for GPU performance measurements.
* **Refreshes.** A refresh step (copy to the CPU plus `refresh()`) costs about 50 ms in the default case. That is
  1 s in total, or 0.6 % of the run.
* **Memory.** The pull sweep caches one temporary PDF field per block pointer. Those of deleted blocks are not freed,
  which is a small leak per refresh.

## Requirements

* waLBerla with AMR support for `lbm_generated` on GPUs: commit `0fd9b5ae0` ("Enable adaptive (dynamic) mesh
  refinement on GPU", branch `suffa/amr_gpu`, not yet on master) on top of the CPU requirements.
  `apps/CMakeLists.txt` skips the app otherwise.
* A build with `WALBERLA_BUILD_WITH_CUDA=ON`.
* pystencils/lbmpy 1.4.x for the legacy code generation.
