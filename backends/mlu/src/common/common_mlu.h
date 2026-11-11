#pragma once

// 对应 InfiniTrain include/common/cuda/common_cuda.h。
// NOTE(MLU-VERIFY): CNRT/CNNL/CNCL 的函数名以服务器上 Neuware 头文件
// （${NEUWARE_HOME}/include/cnrt.h、cnnl.h、cncl.h）为准核一遍。

#include <cnrt.h>
#ifdef USE_CNNL
#include <cnnl.h>
#endif
#ifdef USE_CNCL
#include <cncl.h>
#endif

#include "glog/logging.h"

namespace infini_train::common::mlu {

#define CNRT_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cnrtRet_t status = (call);                                                                                     \
        if (status != cnrtSuccess) {                                                                                   \
            LOG(FATAL) << "CNRT Error: " << cnrtGetErrorStr(status) << " at " << __FILE__ << ":" << __LINE__;          \
        }                                                                                                              \
    } while (0)

#ifdef USE_CNNL
#define CNNL_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cnnlStatus_t status = (call);                                                                                  \
        if (status != CNNL_STATUS_SUCCESS) {                                                                           \
            LOG(FATAL) << "CNNL Error: " << cnnlGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)
#endif

#ifdef USE_CNCL
// cncl.h 自带了一个同名的 CNCL_CHECK 宏（printf 版本），这里先取消它的定义，
// 换成走 glog LOG(FATAL) 的版本，与本后端其它 *_CHECK 宏保持一致的错误处理风格。
#ifdef CNCL_CHECK
#undef CNCL_CHECK
#endif
#define CNCL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        cnclResult_t _status = (expr);                                                                                 \
        if (_status != CNCL_RET_SUCCESS) {                                                                             \
            LOG(FATAL) << "CNCL error: " << cnclGetErrorStr(_status) << " at " << __FILE__ << ":" << __LINE__ << " ("  \
                       << #expr << ")";                                                                                \
        }                                                                                                              \
    } while (0)
#endif

} // namespace infini_train::common::mlu
