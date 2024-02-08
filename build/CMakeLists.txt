cmake_minimum_required(VERSION 3.0.0)
project(cvtest VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(cvtest main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cvtest ${OpenCV_LIBS} $(CMAKE_THREAD_LIBS_INIT))