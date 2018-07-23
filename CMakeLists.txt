cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MACOSX_RPATH 1) # This is needed in macOS

project(BA_Learn)

set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Wall -Wformat")

set(OpenCV_DIR ~/.local/OpenCV34/share/OpenCV)
find_package(OpenCV REQUIRED)
IF (OpenCV_FOUND)
  message([message] OpenCV_VER:${OpenCV_VERSION})
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIBRARY_DIRS})
  add_definitions(${OpenCV_DEFINITIONS})
ENDIF()

add_executable(
  ba_example
  ba_example.cc
  BundleAdjustment2Viewes.cpp
  )

target_link_libraries(
  ba_example
  ${OpenCV_LIBS}
  )
