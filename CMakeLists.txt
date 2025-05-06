cmake_minimum_required(VERSION 3.14)
project(proiect1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} src)

add_executable(proiect1
        main.cpp
        src/proiect1.cpp
        src/proiect1.h
        src/proiect1.cpp
        src/proiect1.h
)

target_link_libraries(proiect1 ${OpenCV_LIBS})
