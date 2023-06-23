add_library(openblas_impl openblas_impl.cpp)
target_include_directories(openblas_impl PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/OpenBLAS)
target_include_directories(openblas_impl PUBLIC ${CMAKE_BINARY_DIR}/3rdparty/OpenBLAS)
# target_link_directories(openblas_impl PUBLIC ${CMAKE_BINARY_DIR}/3rdparty/OpenBLAS/lib)
target_compile_options(openblas_impl PRIVATE -mavx2 -mavx512f -mavx512vl -mavx512dq ${OPENMP_CXX_FLAGS})
add_dependencies(openblas_impl openblas_static)
target_link_libraries(openblas_impl PRIVATE avxop_impl openblas OpenMP::OpenMP_CXX)

target_link_libraries(graph_benchmark openblas_impl)


