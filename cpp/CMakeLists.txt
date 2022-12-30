cmake_minimum_required(VERSION 3.10)
project(RBTree)

include_directories(.)

add_library(library SHARED
  RBTree.cpp
)

add_library(library-python SHARED
  library-python.cpp
)
target_link_libraries(library-python library)
