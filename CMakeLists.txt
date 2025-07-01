cmake_minimum_required(VERSION 3.16)
project(AttentionAllocationTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Kokkos
find_package(Kokkos REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/build)

# Create the test executable
add_executable(test_attention test_attention_main.cpp)
target_link_libraries(test_attention Kokkos::kokkos)
target_include_directories(test_attention PRIVATE kokkos/core/unit_test)

# Create the demo executable
add_executable(demo_attention demo_attention.cpp)
target_link_libraries(demo_attention Kokkos::kokkos)