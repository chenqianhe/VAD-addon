#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::frontend::onnx" for configuration "Debug"
set_property(TARGET openvino::frontend::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::frontend::onnx PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_onnx_frontendd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_onnx_frontendd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::onnx )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::onnx "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_onnx_frontendd.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_onnx_frontendd.dll" )

# Import target "openvino::frontend::paddle" for configuration "Debug"
set_property(TARGET openvino::frontend::paddle APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::frontend::paddle PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_paddle_frontendd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_paddle_frontendd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::paddle )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::paddle "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_paddle_frontendd.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_paddle_frontendd.dll" )

# Import target "openvino::frontend::tensorflow" for configuration "Debug"
set_property(TARGET openvino::frontend::tensorflow APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::frontend::tensorflow PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_tensorflow_fed.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_tensorflow_fed.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::frontend::tensorflow )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::frontend::tensorflow "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_tensorflow_fed.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_tensorflow_fed.dll" )

# Import target "openvino::runtime::c" for configuration "Debug"
set_property(TARGET openvino::runtime::c APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::runtime::c PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_cd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_cd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime::c )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime::c "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvino_cd.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvino_cd.dll" )

# Import target "openvino::runtime" for configuration "Debug"
set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(openvino::runtime PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvinod.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvinod.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvino::runtime )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvino::runtime "${_IMPORT_PREFIX}/runtime/lib/intel64/Debug/openvinod.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/Debug/openvinod.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
