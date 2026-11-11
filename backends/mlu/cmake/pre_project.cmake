# Cambricon MLU provider: settings that must be known before project().
#
# The host compiler stays whatever the user selected (the framework needs C++20,
# GCC 13 in the validated environment). BANG C device sources are compiled by
# cncc through a custom command in this provider's CMakeLists, so no compiler
# replacement happens here.

set(_MLU_DEFAULT_NEUWARE "$ENV{NEUWARE_HOME}")
if(NOT _MLU_DEFAULT_NEUWARE)
  set(_MLU_DEFAULT_NEUWARE "/usr/local/neuware")
endif()
set(NEUWARE_HOME "${_MLU_DEFAULT_NEUWARE}" CACHE PATH
  "Cambricon Neuware SDK root (CNRT, cncc and optional CNNL/CNCL)")
unset(_MLU_DEFAULT_NEUWARE)

if(NOT EXISTS "${NEUWARE_HOME}/bin/cncc")
  message(FATAL_ERROR
    "The BANG C compiler was not found at ${NEUWARE_HOME}/bin/cncc. "
    "Set NEUWARE_HOME or pass -DNEUWARE_HOME=<sdk-root>.")
endif()
if(NOT EXISTS "${NEUWARE_HOME}/include/cnrt.h")
  message(FATAL_ERROR
    "CNRT headers were not found at ${NEUWARE_HOME}/include/cnrt.h.")
endif()

# The framework is built for its CPU backend; CUDA/NCCL belong to other
# providers. MLU multi-card collectives are provided by CNCL (the CnclImpl in
# backends/mlu/src/core/ccl/), not by CUDA/NCCL, so USE_NCCL stays OFF here.
set(USE_CUDA OFF)
set(USE_NCCL OFF)
set(BUILD_SHARED_LIBS OFF)
