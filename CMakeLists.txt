cmake_minimum_required(VERSION 3.13)

project(china-graph-challenge)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# add ccache
find_program(CCACHE_FOUND NAMES ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    # set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# add benchmark library
set(BENCHMARK_DOWNLOAD_DEPENDENCIES True)
add_subdirectory(3rdparty/benchmark)

# add PaRMAT library
set(PaRMAT_DIR 3rdparty/PaRMAT)
file(GLOB PaRMAT_FILES "${PaRMAT_DIR}/src/*.cpp")
add_executable(PaRMAT ${PaRMAT_FILES})
target_link_libraries(PaRMAT PRIVATE pthread)

# add OpenBLAS library
add_subdirectory(3rdparty/OpenBLAS)

#add include dirs
include_directories(${CMAKE_SOURCE_DIR}/include)

# add utils
add_subdirectory(utils)

# add benchmark
add_subdirectory(benchmark)

# add impls
add_subdirectory(impls)