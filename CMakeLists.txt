cmake_minimum_required(VERSION 3.21)
project(Map)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")

find_package(OpenCV REQUIRED)

add_executable(Map main.cpp)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Map
  ${OpenCV_LIBRARIES}
  )


