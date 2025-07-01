kp_add_library(kp_roctx_connector kp_roctx_connector.cpp)

# As of ROCm 6.2, it is recommended to use ROCtx provided by rocprofiler-sdk-roctx
# instead of the "old" one provided by roctracer.
#
# See also: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-rocprofv3.html
find_package(rocprofiler-sdk-roctx CONFIG PATHS $ENV{ROCM_PATH})
if(rocprofiler-sdk-roctx_FOUND)
    target_link_libraries(kp_roctx_connector PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
else()
    find_library(ROCM_ROCTX_LIB roctx64 REQUIRED HINTS $ENV{ROCM_PATH}/lib)
    find_path(ROCM_ROCTX_INCLUDE roctx.h REQUIRED HINTS $ENV{ROCM_PATH}/include/roctracer)

    target_include_directories(kp_roctx_connector PRIVATE ${ROCM_ROCTX_INCLUDE})
    target_link_libraries(kp_roctx_connector PRIVATE ${ROCM_ROCTX_LIB})
endif()
