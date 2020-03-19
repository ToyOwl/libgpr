cmake_minimum_required(VERSION 3.8.0)

project(libgpr)

cmake_policy(SET CMP0016 NEW)
cmake_policy(SET CMP0057 NEW)

set(CMAKE_CXX_STANDARD 14)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
if(MSVC)

  if(MSVC_VERSION VERSION_LESS 1900)
    message(FATAL_ERROR "${PROJECT_NAME} requires VS2015 or greater.")
  endif()

elseif(CMAKE_COMPILER_IS_GNUCXX)

  set(CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3.1)
    message(FATAL_ERROR "The installed g++ version is ${CMAKE_CXX_COMPILER_VERSION}. ${PROJECT_NAME} requires g++ 5.3.1 or greater.")
  endif()
endif()


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(NLopt REQUIRED)
find_package(Eigen3 REQUIRED)

 
include_directories(${EIGEN3_INCLUDE_DIR}   ${EIGEN3_ROOT_DIR} ${NLOPT_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

SET(SOURCES src/GaussianProcessRegressionTest.cc)
SET(HEADERS kernels/KernelBase.hh  
            kernels/RationalQuadratic.hh
            kernels/MaternThreeHalves.hh 
            kernels/MaternThreeHalvesPeriodic.hh  
            kernels/MaternFiveHalves.hh
            kernels/RationalQuadratic.hh
            kernels/RadialBasisKernel.hh
            optimization/KernelOptimizator.hh
            optimization/GradientBasedKernelOptimizer.hh
            optimization/DirectBasedKernelOptimizer.hh
            models/GaussianProcessRegressionModels.hh)

add_executable(libgprtest ${SOURCES} ${HEADERS})
target_link_libraries(libgprtest ${NLOPT_STATIC})

