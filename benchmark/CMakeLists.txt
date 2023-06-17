add_executable(graph_benchmark benchmark.cpp)

target_include_directories(graph_benchmark PUBLIC impls utils)

target_link_libraries(graph_benchmark benchmark::benchmark utils)