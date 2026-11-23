# Adaptive mesh refinement benchmark (GPU)

GPU version of [`BenchmarkApp_cpu_adaptiveRefinement`](../BenchmarkApp_cpu_adaptiveRefinement/README.md). It runs the
same case with the same refinement criterion, the same `input.prm` parameters and the same output, so see the CPU
README for how the adaptive refinement works and for the output. The kernels are those of
`BenchmarkApp_gpu_staticRefinement`: cumulant D3Q27 with QR subgrid model and Guo forcing, fast divisions and square
roots, and a compile-time GPU block size of (128, 1, 1). One difference: the streaming pattern is pull, not AA (see
below).

## Running

Use one MPI process per GPU:

```
mpirun -np 1 ./BenchmarkApp_gpu_adaptive_refinement input.prm
```

**Several GPUs per node:** give every process only its own GPU:

```
mpirun -np 2 sh -c 'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK exec ./BenchmarkApp_gpu_adaptive_refinement input.prm'
```

* With the GTX 980 Ti (Maxwell) on this machine, a process whose current device is not device 0 computes a NaN
  turbine AABB in `GPUTurbineTopology::getAABB()`. That process then builds an unrefined forest, and the app stops
  with "The initial block forest must contain blocks on the finest level".
* The cause is probably the `__managed__` globals of the GPU turbine topology on pre-Pascal GPUs. It is not specific
  to this app and is not investigated further.
* With the command above, waLBerla prints a harmless warning ("Too many processes started per node").

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

Measured on GTX 980 Ti GPUs in single precision, with the default case, `refinementFrequency 50` and 1000 time steps:

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
