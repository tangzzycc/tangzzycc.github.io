

add_executable(my_reduce_v0_global_memory my_reduce_v0_global_memory.cu)
target_link_libraries(my_reduce_v0_global_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_reduce_v1_shared_memory my_reduce_v1_shared_memory.cu)
target_link_libraries(my_reduce_v1_shared_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v1_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()



add_executable(my_reduce_v2_no_divergence_branch my_reduce_v2_no_divergence_branch.cu)
target_link_libraries(my_reduce_v2_no_divergence_branch PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v2_no_divergence_branch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_reduce_v3_no_bank_conflict my_reduce_v3_no_bank_conflict.cu)
target_link_libraries(my_reduce_v3_no_bank_conflict PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v3_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v4_add_during_load_plan_a my_reduce_v4_add_during_load_plan_a.cu)
target_link_libraries(my_reduce_v4_add_during_load_plan_a PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v4_add_during_load_plan_a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_reduce_v4_add_during_load_plan_b my_reduce_v4_add_during_load_plan_b.cu)
target_link_libraries(my_reduce_v4_add_during_load_plan_b PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v4_add_during_load_plan_b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_reduce_v5_unroll_last_warp my_reduce_v5_unroll_last_warp.cu)
target_link_libraries(my_reduce_v5_unroll_last_warp PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v5_unroll_last_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_reduce_v6_completely_unroll my_reduce_v6_completely_unroll.cu)
target_link_libraries(my_reduce_v6_completely_unroll PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v6_completely_unroll PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_reduce_v7_mutli_add my_reduce_v7_mutli_add.cu)
target_link_libraries(my_reduce_v7_mutli_add PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v7_mutli_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v8_shuffle my_reduce_v8_shuffle.cu)
target_link_libraries(my_reduce_v8_shuffle PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v8_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

