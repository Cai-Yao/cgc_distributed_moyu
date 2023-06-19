add_library(openblas_impl openblas_impl.cpp)
target_include_directories(openblas_impl PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/OpenBLAS)
target_include_directories(openblas_impl PUBLIC ${CMAKE_BINARY_DIR}/3rdparty/OpenBLAS)
# target_link_directories(openblas_impl PUBLIC ${CMAKE_BINARY_DIR}/3rdparty/OpenBLAS/lib)
target_link_libraries(openblas_impl OpenBLAS)
add_dependencies(openblas_impl openblas_static)

target_link_libraries(graph_benchmark openblas_impl)


