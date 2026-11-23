
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( airfoils )
waLBerla_link_files_to_builddir( *.blade )
waLBerla_link_files_to_builddir( *.tower )

waLBerla_generate_target_from_python(
        NAME BenchmarkApp_cpu_adaptive_refinement_Generation
        FILE LBMGeneration.py
        OUT_FILES
        waLBerlaWind_KernelInfo.h
        waLBerlaWind_SweepCollection.cpp waLBerlaWind_SweepCollection.h
        waLBerlaWind_StorageSpecification.cpp waLBerlaWind_StorageSpecification.h
        waLBerlaWind_BoundaryCollection.h
        waLBerlaWind_NoSlip.cpp waLBerlaWind_NoSlip.h
        waLBerlaWind_FreeSlip.cpp waLBerlaWind_FreeSlip.h
        waLBerlaWind_WFB.cpp waLBerlaWind_WFB.h
        waLBerlaWind_UniformUBB.cpp waLBerlaWind_UniformUBB.h
        waLBerlaWind_LogLawUBB.cpp waLBerlaWind_LogLawUBB.h
        waLBerlaWind_Outflow.cpp waLBerlaWind_Outflow.h
        waLBerlaWind_PdfSetter.cpp waLBerlaWind_PdfSetter.h
        FlowDriverCollection.h
        Drivers.h Drivers.cpp
)

add_executable( BenchmarkApp_cpu_adaptive_refinement main.cpp )
target_link_libraries( BenchmarkApp_cpu_adaptive_refinement
        PRIVATE walberla::core walberla::blockforest walberla::domain_decomposition walberla::field walberla::lbm
                walberla::timeloop
                BenchmarkApp_cpu_adaptive_refinement_Generation
                domain component turbine_topology force_model
                wind_turbine_core
)
