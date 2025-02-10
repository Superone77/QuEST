if (CUDA_FOUND)

endif ()

set(_C_LIBRARIES "")

add_subdirectory(symmetric)

pybind11_add_module(_C MODULE THIN_LTO binding.cpp)

target_link_libraries(
        _C PUBLIC
        ${TORCH_LIBRARIES}
        ${_C_LIBRARIES}
)
