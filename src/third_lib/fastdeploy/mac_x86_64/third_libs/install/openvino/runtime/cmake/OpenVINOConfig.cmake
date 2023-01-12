# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# FindOpenVINO
# ------
#
# Provides OpenVINO runtime for model creation and inference, frontend libraries
# to convert models from framework specific formats.
#
# The following components are supported:
#
#  * `Runtime`: OpenVINO C++ and C Core & Inference Runtime, frontend common
#  * `ONNX`: OpenVINO ONNX frontend
#  * `Paddle`: OpenVINO Paddle frontend
#  * `TensorFlow`: OpenVINO TensorFlow frontend
#
# If no components are specified, `Runtime` component is provided:
#
#   find_package(OpenVINO REQUIRED) # only Runtime component
#
# If specific components are required:
#
#   find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
#
# Imported Targets:
# ------
#
#  Runtime targets:
#
#   `openvino::runtime`
#   The OpenVINO C++ Core & Inference Runtime
#
#   `openvino::runtime::c`
#   The OpenVINO C Inference Runtime
#
#  Frontend specific targets:
#
#   `openvino::frontend::onnx`
#   ONNX FrontEnd target (optional)
#
#   `openvino::frontend::paddle`
#   Paddle FrontEnd target (optional)
#
#   `openvino::frontend::tensorflow`
#   TensorFlow FrontEnd target (optional)
#
# Result variables:
# ------
#
# The module sets the following variables in your project:
#
#   `OpenVINO_FOUND`
#   System has OpenVINO Runtime installed
#
#   `OpenVINO_Runtime_FOUND`
#   OpenVINO C++ Core & Inference Runtime is available
#
#   `OpenVINO_Frontend_ONNX_FOUND`
#   OpenVINO ONNX frontend is available
#
#   `OpenVINO_Frontend_Paddle_FOUND`
#   OpenVINO Paddle frontend is available
#
#   `OpenVINO_Frontend_TensorFlow_FOUND`
#   OpenVINO TensorFlow frontend is available
#
#   `OpenVINO_Frontend_IR_FOUND`
#   OpenVINO IR frontend is available
#
#  OpenVINO version variables:
#
#   `OpenVINO_VERSION_MAJOR`
#   Major version component
# 
#   `OpenVINO_VERSION_MINOR`
#   minor version component
#
#   `OpenVINO_VERSION_PATCH`
#   Patch version component
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was OpenVINOConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

#
# Common functions
#

if(NOT DEFINED CMAKE_FIND_PACKAGE_NAME)
    set(CMAKE_FIND_PACKAGE_NAME OpenVINO)
    set(_need_package_name_reset ON)
endif()

# we have to use our own version of find_dependency because of support cmake 3.7
macro(_ov_find_dependency dep)
    set(cmake_fd_quiet_arg)
    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        set(cmake_fd_quiet_arg QUIET)
    endif()
    set(cmake_fd_required_arg)
    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        set(cmake_fd_required_arg REQUIRED)
    endif()

    get_property(cmake_fd_alreadyTransitive GLOBAL PROPERTY
        _CMAKE_${dep}_TRANSITIVE_DEPENDENCY)

    find_package(${dep} ${ARGN}
        ${cmake_fd_quiet_arg}
        ${cmake_fd_required_arg})

    if(NOT DEFINED cmake_fd_alreadyTransitive OR cmake_fd_alreadyTransitive)
        set_property(GLOBAL PROPERTY _CMAKE_${dep}_TRANSITIVE_DEPENDENCY TRUE)
    endif()

    if(NOT ${dep}_FOUND)
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${CMAKE_FIND_PACKAGE_NAME} could not be found because dependency ${dep} could not be found.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND False)
        return()
    endif()

    set(cmake_fd_required_arg)
    set(cmake_fd_quiet_arg)
endmacro()

function(_ov_target_no_deprecation_error)
    if(NOT MSVC)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(flags "-diag-warning=1786")
        else()
            set(flags "-Wno-error=deprecated-declarations")
        endif()
        if(CMAKE_CROSSCOMPILING)
            set_target_properties(${ARGV} PROPERTIES
                                  INTERFACE_LINK_OPTIONS "-Wl,--allow-shlib-undefined")
        endif()

        set_target_properties(${ARGV} PROPERTIES INTERFACE_COMPILE_OPTIONS ${flags})
    endif()
endfunction()

#
# OpenVINO config
#

# need to store current PACKAGE_PREFIX_DIR, because it's overwritten by sub-package one
set(_ov_package_prefix_dir "${PACKAGE_PREFIX_DIR}")

set(THREADING "TBB")
if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO") AND NOT TBB_FOUND)
    set(enable_system_tbb "OFF")
    if(NOT enable_system_tbb)
        set_and_check(_tbb_dir "${PACKAGE_PREFIX_DIR}/runtime/3rdparty/tbb/")

        # see https://stackoverflow.com/questions/28070810/cmake-generate-error-on-windows-as-it-uses-as-escape-seq
        if(DEFINED ENV{TBBROOT})
            file(TO_CMAKE_PATH $ENV{TBBROOT} ENV_TBBROOT)
        endif()
        if(DEFINED ENV{TBB_DIR})
            file(TO_CMAKE_PATH $ENV{TBB_DIR} ENV_TBB_DIR)
        endif()

        set(find_package_tbb_extra_args
            CONFIG
            PATHS
                # oneTBB case exposed via export TBBROOT=<custom TBB root>
                "${ENV_TBBROOT}/lib64/cmake/TBB"
                "${ENV_TBBROOT}/lib/cmake/TBB"
                "${ENV_TBBROOT}/lib/cmake/tbb"
                "${ENV_TBB_DIR}"
                # for custom TBB exposed via cmake -DTBBROOT=<custom TBB root>
                "${TBBROOT}/cmake"
                # _tbb_dir points to TBB_DIR (custom | temp | system) used to build OpenVINO
                ${_tbb_dir}
            NO_CMAKE_FIND_ROOT_PATH
            NO_DEFAULT_PATH)
        unset(_tbb_dir)
    endif()
    unset(enable_system_tbb)

    _ov_find_dependency(TBB
                        COMPONENTS tbb tbbmalloc
                        ${find_package_tbb_extra_args})

    set(install_tbbbind "")
    if(install_tbbbind)
        set_and_check(_tbb_bind_dir "")
        _ov_find_dependency(TBBBIND_2_5
                            PATHS ${_tbb_bind_dir}
                            NO_CMAKE_FIND_ROOT_PATH
                            NO_DEFAULT_PATH)
        set_target_properties(${TBBBIND_2_5_IMPORTED_TARGETS} PROPERTIES IMPORTED_GLOBAL ON)
    endif()
    unset(install_tbbbind)
endif()

_ov_find_dependency(Threads)

set(ENABLE_INTEL_GNA "OFF")
set(ENABLE_INTEL_GNA_SHARED "ON")
if(ENABLE_INTEL_GNA AND NOT ENABLE_INTEL_GNA_SHARED AND NOT libGNA_FOUND)
    set_and_check(GNA_PATH "")
    _ov_find_dependency(libGNA
                        COMPONENTS KERNEL
                        CONFIG
                        PATHS "${CMAKE_CURRENT_LIST_DIR}"
                        NO_CMAKE_FIND_ROOT_PATH
                        NO_DEFAULT_PATH)
endif()

if(NOT TARGET openvino)
    set(_ov_as_external_package ON)
    include("${CMAKE_CURRENT_LIST_DIR}/OpenVINOTargets.cmake")

    # WA for cmake version < 3.16 which does not export
    # IMPORTED_LINK_DEPENDENT_LIBRARIES_** properties if no PUBLIC dependencies for the library
    if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO") AND TBB_FOUND)
        foreach (type RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
            set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_LINK_DEPENDENT_LIBRARIES_${type} "TBB::tbb;TBB::tbbmalloc")
        endforeach()
    endif()
endif()

#
# Components
#

set(${CMAKE_FIND_PACKAGE_NAME}_Runtime_FOUND ON)

set(${CMAKE_FIND_PACKAGE_NAME}_ONNX_FOUND ON)
set(${CMAKE_FIND_PACKAGE_NAME}_Paddle_FOUND ON)
set(${CMAKE_FIND_PACKAGE_NAME}_TensorFlow_FOUND ON)
set(${CMAKE_FIND_PACKAGE_NAME}_IR_FOUND ON)

set(${CMAKE_FIND_PACKAGE_NAME}_Frontend_ONNX_FOUND ${${CMAKE_FIND_PACKAGE_NAME}_ONNX_FOUND})
set(${CMAKE_FIND_PACKAGE_NAME}_Frontend_Paddle_FOUND ${${CMAKE_FIND_PACKAGE_NAME}_Paddle_FOUND})
set(${CMAKE_FIND_PACKAGE_NAME}_Frontend_TensorFlow_FOUND ${${CMAKE_FIND_PACKAGE_NAME}_TensorFlow_FOUND})
set(${CMAKE_FIND_PACKAGE_NAME}_Frontend_IR_FOUND ${${CMAKE_FIND_PACKAGE_NAME}_IR_FOUND})

# if no components specified, only Runtime is provided
if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    set(${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS Runtime)
endif()

#
# Apply common functions
#

foreach(target openvino::runtime openvino::runtime::c
               openvino::frontend::onnx openvino::frontend::paddle openvino::frontend::tensorflow)
    if(TARGET ${target} AND _ov_as_external_package)
        _ov_target_no_deprecation_error(${target})
    endif()
endforeach()
unset(_ov_as_external_package)

# restore PACKAGE_PREFIX_DIR
set(PACKAGE_PREFIX_DIR ${_ov_package_prefix_dir})
unset(_ov_package_prefix_dir)

check_required_components(${CMAKE_FIND_PACKAGE_NAME})

if(_need_package_name_reset)
    unset(CMAKE_FIND_PACKAGE_NAME)
    unset(_need_package_name_reset)
endif()

unset(${CMAKE_FIND_PACKAGE_NAME}_IR_FOUND)
unset(${CMAKE_FIND_PACKAGE_NAME}_Paddle_FOUND)
unset(${CMAKE_FIND_PACKAGE_NAME}_ONNX_FOUND)
unset(${CMAKE_FIND_PACKAGE_NAME}_TensorFlow_FOUND)
