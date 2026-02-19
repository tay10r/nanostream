cmake_minimum_required(VERSION 3.20)

project(nanostream)

option(NANOSTREAM_EVAL "Build the evaluation program." OFF)

add_library(nanostream
  nanostream.h
  nanostream.c
  nanostream_eigen.c
)

target_include_directories(nanostream PUBLIC .)

if(NANOSTREAM_EVAL)
  add_executable(eval
    eval/main.cpp
    eval/stb_image.h
    eval/stb_image.c
    eval/stb_image_write.h
    eval/stb_image_write.c
  )
  target_link_libraries(eval
    PUBLIC
      nanostream
  )
endif()
