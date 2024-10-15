cmake_minimum_required(VERSION 3.10)

project(hash)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(${PROJECT_NAME} ${SRC_FILES})
include_directories(include)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})