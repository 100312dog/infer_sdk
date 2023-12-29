cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(infer_sdk LANGUAGES CXX CUDA)
add_definitions(-DTENSORRT_BACKEND)

set(CWD ${CMAKE_CURRENT_LIST_DIR})

set(TensorRT_DIR "/usr/local/TensorRT-8.5.3.1")
file(GLOB TensorRT_LIBS ${TensorRT_DIR}/lib/*.so)
set(TensorRT_INCLUDE_DIRS ${TensorRT_DIR}/include)


find_package(CUDA 10.2 REQUIRED)
find_package(OpenCV REQUIRED PATHS ${CWD}/third_party/opencv_3.2.0_x64 NO_DEFAULT_PATH)

add_subdirectory(${CWD}/third_party/spdlog_1.12.0_emb)
add_subdirectory(${CWD}/third_party/eigen)


find_package(Threads REQUIRED)
# find_package(spdlog REQUIRED PATHS ${CWD}/third_party/spdlog_1.12.0_pre NO_DEFAULT_PATH)

set(INCLUDE_DIRS
        ${CWD}/include
        ${CWD}/third_party/nlohmann
        ${CWD}/third_party/tclap_1.2.5/include
        ${CWD}/third_party/byte_track/include
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        # ${CWD}/third_party/spdlog_1.12.0/include
        )

include_directories(
    ${INCLUDE_DIRS}
)

set(LINK_LIBS
        Threads::Threads
        spdlog::spdlog
        Eigen3::Eigen
        # spdlog::spdlog_header_only
        ${TensorRT_LIBS}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        )

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        ${CWD}/src/*.cc
        ${CWD}/third_party/byte_track/src/*.cpp)

message(STATUS ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})






