cmake_minimum_required(VERSION 2.8)
project(calicam_mono LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

add_executable(calicam_mono calicam_mono.cpp)
target_link_libraries(calicam_mono ${OpenCV_LIBRARIES})
