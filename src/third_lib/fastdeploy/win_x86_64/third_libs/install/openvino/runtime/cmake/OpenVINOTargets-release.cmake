#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::frontend::onnx" for configuration "Release"
set_property(TARGET openvino::frontend::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::onnx PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_onnx_frontend.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_onnx_frontend.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::onnx )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::onnx "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_onnx_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_onnx_frontend.dll" )

# Import target "openvino::frontend::paddle" for configuration "Release"
set_property(TARGET openvino::frontend::paddle APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::paddle PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_paddle_frontend.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_paddle_frontend.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::paddle )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::paddle "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_paddle_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_paddle_frontend.dll" )

# Import target "openvino::frontend::tensorflow" for configuration "Release"
set_property(TARGET openvino::frontend::tensorflow APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::frontend::tensorflow PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_tensorflow_fe.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_tensorflow_fe.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::tensorflow )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::tensorflow "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_tensorflow_fe.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_tensorflow_fe.dll" )

# Import target "openvino::runtime::c" for configuration "Release"
set_property(TARGET openvino::runtime::c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::runtime::c PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_c.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_c.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime::c )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime::c "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino_c.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino_c.dll" )

# Import target "openvino::runtime" for configuration "Release"
set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvino::runtime PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime "${_IMPORT_PREFIX}/runtime/lib/intel64/Release/openvino.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Release/openvino.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
