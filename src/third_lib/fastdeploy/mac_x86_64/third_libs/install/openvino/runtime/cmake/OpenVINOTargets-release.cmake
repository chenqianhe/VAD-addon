#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::frontend::onnx" for configuration "Release"
set_property(TARGET openvino::frontend::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::onnx PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_onnx_frontend.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopenvino_onnx_frontend.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::onnx )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::onnx "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_onnx_frontend.dylib" )

# Import target "openvino::frontend::paddle" for configuration "Release"
set_property(TARGET openvino::frontend::paddle APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::paddle PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_paddle_frontend.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopenvino_paddle_frontend.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::paddle )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::paddle "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_paddle_frontend.dylib" )

# Import target "openvino::frontend::tensorflow" for configuration "Release"
set_property(TARGET openvino::frontend::tensorflow APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::tensorflow PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_tensorflow_fe.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopenvino_tensorflow_fe.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::tensorflow )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::tensorflow "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_tensorflow_fe.dylib" )

# Import target "openvino::runtime::c" for configuration "Release"
set_property(TARGET openvino::runtime::c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::runtime::c PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "openvino::runtime"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_c.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopenvino_c.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime::c )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime::c "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino_c.dylib" )

# Import target "openvino::runtime" for configuration "Release"
set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::runtime PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "TBB::tbb;TBB::tbbmalloc"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopenvino.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/libopenvino.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
