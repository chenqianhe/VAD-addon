#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::frontend::onnx" for configuration "Debug"
set_property(TARGET openvino::frontend::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::frontend::onnx PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_onnx_frontendd.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libopenvino_onnx_frontendd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::onnx )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::onnx "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_onnx_frontendd.dylib" )

# Import target "openvino::frontend::paddle" for configuration "Debug"
set_property(TARGET openvino::frontend::paddle APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::frontend::paddle PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_paddle_frontendd.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libopenvino_paddle_frontendd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::paddle )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::paddle "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_paddle_frontendd.dylib" )

# Import target "openvino::frontend::tensorflow" for configuration "Debug"
set_property(TARGET openvino::frontend::tensorflow APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::frontend::tensorflow PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_tensorflow_fed.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libopenvino_tensorflow_fed.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::tensorflow )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::tensorflow "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_tensorflow_fed.dylib" )

# Import target "openvino::runtime::c" for configuration "Debug"
set_property(TARGET openvino::runtime::c APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::runtime::c PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "openvino::runtime"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_cd.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libopenvino_cd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime::c )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime::c "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvino_cd.dylib" )

# Import target "openvino::runtime" for configuration "Debug"
set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::runtime PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "TBB::tbb;TBB::tbbmalloc"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvinod.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libopenvinod.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/libopenvinod.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
