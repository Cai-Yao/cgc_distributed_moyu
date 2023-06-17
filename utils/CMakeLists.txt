add_library(utils utils.cpp)

file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/data)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/data/graph)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/data/weight)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/data/embedding)

add_dependencies(utils PaRMAT)