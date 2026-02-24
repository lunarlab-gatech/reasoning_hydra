# Python bindings for the HydraMeshMapper facade.
# Requires pybind11 >= 2.6 and the reasoning_hydra C++ library to be compiled.
#
# Build the extension with:
#   cmake -DHYDRA_ENABLE_PYTHON=ON ..
#   make hydra_python

find_package(pybind11 REQUIRED)

pybind11_add_module(hydra_python bindings.cpp
    ../src/python/mesh_mapper.cpp)

target_include_directories(hydra_python PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS})

target_link_libraries(hydra_python PRIVATE
    reasoning_hydra
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Eigen3::Eigen)

# Install the .so next to the rest of the Python package so it can be
# imported as `import hydra_python`.
install(TARGETS hydra_python DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python)
