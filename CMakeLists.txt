cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(BA_Learn)

set(CMAKE_MACOSX_RPATH 1) # This is needed in macOS
set( CMAKE_BUILD_TYPE Release )


IF (!APPLE)
  set(CMAKE_CXX_COMPILER g++)
ENDIF()

set( CMAKE_CXX_FLAGS "-std=c++11" )
find_package( Eigen3 REQUIRED )

set(OpenCV_DIR "~/.local/OpenCV/share/OpenCV")
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

