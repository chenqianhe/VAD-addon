# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)

if(THIRD_PARTY_PATH)
  SET(GFLAGS_PREFIX_DIR  ${THIRD_PARTY_PATH}/gflags)
  SET(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)
else()
  # For example cmake
  SET(GFLAGS_PREFIX_DIR  ${FASTDEPLOY_INSTALL_DIR}/installed_fastdeploy/cmake)
  SET(GFLAGS_INSTALL_DIR ${FASTDEPLOY_INSTALL_DIR}/installed_fastdeploy/cmake/gflags)
endif()
SET(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
set(GFLAGS_SOURCE_FILE ${GFLAGS_PREFIX_DIR}/src/gflags.tgz CACHE PATH "gflags source file." FORCE)

set(GFLAGS_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
set(GFLAGS_URL ${GFLAGS_URL_PREFIX}/gflags.tgz)
set(GFLAGS_CACHE_FILE ${CMAKE_CURRENT_LIST_DIR}/gflags.tgz)
if(EXISTS ${GFLAGS_CACHE_FILE})
  set(GFLAGS_URL ${GFLAGS_CACHE_FILE} CACHE PATH "gflags cache file." FORCE)
  set(GFLAGS_SOURCE_FILE ${GFLAGS_CACHE_FILE} CACHE PATH "gflags source file." FORCE)
endif()

IF(WIN32)
  set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/gflags_static.lib" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
ELSE(WIN32)
  set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
  set(BUILD_COMMAND $(MAKE) --silent)
  set(INSTALL_COMMAND $(MAKE) install)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIR})

if(ANDROID)
  ExternalProject_Add(
    extern_gflags
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${GFLAGS_URL}
    PREFIX          ${GFLAGS_PREFIX_DIR}
    UPDATE_COMMAND  ""
    BUILD_COMMAND   ${BUILD_COMMAND}
    INSTALL_COMMAND ${INSTALL_COMMAND}
    CMAKE_ARGS      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                    -DANDROID_ABI=${ANDROID_ABI}
                    -DANDROID_NDK=${ANDROID_NDK}
                    -DANDROID_PLATFORM=${ANDROID_PLATFORM}
                    -DANDROID_STL=c++_static
                    -DANDROID_TOOLCHAIN=${ANDROID_TOOLCHAIN}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DBUILD_STATIC_LIBS=ON
                    -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
   CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}                
    BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES}
)
else()
  ExternalProject_Add(
      extern_gflags
      ${EXTERNAL_PROJECT_LOG_ARGS}
      URL ${GFLAGS_URL}
      PREFIX          ${GFLAGS_PREFIX_DIR}
      UPDATE_COMMAND  ""
      BUILD_COMMAND   ${BUILD_COMMAND}
      INSTALL_COMMAND ${INSTALL_COMMAND}
      CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                      -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                      -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                      -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                      -DBUILD_STATIC_LIBS=ON
                      -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
                      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                      -DBUILD_TESTING=OFF
                      -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                      ${EXTERNAL_OPTIONAL_ARGS}
      CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
                      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES}
  )
endif()
ADD_LIBRARY(gflags STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES})
ADD_DEPENDENCIES(gflags extern_gflags)

if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
  list(APPEND GFLAGS_LIBRARIES pthread)
endif()

# On Windows (including MinGW), the Shlwapi library is used by gflags if available.
if (WIN32)
  include(CheckIncludeFileCXX)
  check_include_file_cxx("shlwapi.h" HAVE_SHLWAPI)
  if (HAVE_SHLWAPI)
    set_property(GLOBAL PROPERTY OS_DEPENDENCY_MODULES shlwapi.lib)
    list(APPEND GFLAGS_LIBRARIES shlwapi.lib)
  endif(HAVE_SHLWAPI)
endif (WIN32)
