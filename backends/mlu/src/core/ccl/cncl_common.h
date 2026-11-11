#pragma once

// 对应 InfiniTrain/src/core/ccl/cuda/nccl_common.h。
// 把框架 CclComm/CclUniqueId 抽象绑定到 CNCL 的 cnclComm_t / cnclCliqueId。
// CnclImpl（见同目录 cncl_impl.cc）只经本头访问底层句柄，不直接碰 <cncl.h>。

#include <cncl.h>

#include "infini_train/include/core/ccl/ccl_common.h"

namespace infini_train::core {

// 持有一个 CNCL communicator 句柄。生命周期由框架的 unique_ptr<CclComm> 管理。
class CnclComm final : public CclComm {
 public:
  CnclComm();
  explicit CnclComm(cnclComm_t comm);

  cnclComm_t cncl_comm() const;
  void set_cncl_comm(cnclComm_t comm);

 private:
  cnclComm_t cncl_comm_ = nullptr;
};

// 持有一个 CNCL clique id（128 字节 + 8 字节 hash，见 cncl.h 的 cnclCliqueId）。
// 单进程多卡 DP 路径下 GetUniqueId() 直接 cnclGetCliqueId 即可；多进程路径会经
// 框架的 WriteUniqueIdFile/ReadUniqueIdFile 在进程间分发本结构体的裸字节。
class CnclUniqueId final : public CclUniqueId {
 public:
  CnclUniqueId();
  explicit CnclUniqueId(const cnclCliqueId &id);

  size_t Size() const override;
  const void *Data() const override;
  void Load(const void *src, size_t size) override;

  cnclCliqueId *cncl_clique_id();
  const cnclCliqueId *cncl_clique_id() const;

 private:
  cnclCliqueId id_{};
};

}  // namespace infini_train::core
