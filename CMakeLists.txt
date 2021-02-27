CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(SIFTGPU C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -Wno-deprecated -fPIC")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-write-strings -Wno-unused-result -Wno-deprecated -fPIC")

find_package(OpenCV REQUIRED)

ADD_SUBDIRECTORY(SiftGPU)

add_executable(maino maino.cpp)
#target_include_directories(maino PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/SiftGPU)
add_dependencies(maino siftgpu)
target_link_libraries(maino
        siftgpu
        ${OpenCV_LIBS}
        )

