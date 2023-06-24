add_library(avxop_impl exp.cpp)
target_compile_options(avxop_impl PRIVATE -mavx2 -mavx512f ${OPENMP_CXX_FLAGS})
target_link_libraries(avxop_impl PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(graph_benchmark avxop_impl)