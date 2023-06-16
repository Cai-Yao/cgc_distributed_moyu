add_library(utils gen_data.cpp)

add_custom_command(
    TARGET utils POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy 
    ${CMAKE_CURRENT_SOURCE_DIR}/PaRMAT 
    ${EXECUTABLE_OUTPUT_PATH})