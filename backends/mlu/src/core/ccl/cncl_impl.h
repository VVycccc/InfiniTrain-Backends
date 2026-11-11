#pragma once

// 对应 InfiniTrain/src/core/ccl/cuda/nccl_impl.h。
// 把框架 core::CclImpl 的 14 个虚函数绑定到 CNCL，注册到 Device::DeviceType::kPrivateUse1。
// 一旦本 TU 被链接进可执行文件（--whole-archive），CclImplRegistry 就有 kPrivateUse1 的条目，
// 框架的 DDP/ProcessGroup 即可在 MLU 上跑集合通信。

#include <cncl.h>

#include "infini_train/include/core/ccl/ccl.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train::core {

class CnclImpl final : public CclImpl {
 public:
  Device::DeviceType Type() const override;

  void GroupStart() const override;
  void GroupEnd() const override;

  void GetAsyncError(const CclComm *comm, CclStatus *async_error) const override;

  void GetUniqueId(CclUniqueId **unique_id) const override;

  void CommInitAll(CclComm **comms, int ndev, const int *devlist) const override;

  void CommInitRank(CclComm **comm, int nranks,
                    const CclUniqueId &unique_id, int rank) const override;

  void CommDestroy(CclComm *comm) const override;

  void AllReduce(const void *sendbuff, void *recvbuff, size_t count,
                 DataType dtype, nn::parallel::comm::ReduceOpType reduce_op,
                 const CclComm *comm, Stream *stream) const override;

  void Broadcast(const void *sendbuff, void *recvbuff, size_t count,
                 DataType dtype, int root, const CclComm *comm,
                 Stream *stream) const override;

  void Reduce(const void *sendbuff, void *recvbuff, size_t count,
              DataType dtype, nn::parallel::comm::ReduceOpType reduce_op,
              int root, const CclComm *comm, Stream *stream) const override;

  void AllGather(const void *sendbuff, void *recvbuff, size_t count,
                 DataType dtype, const CclComm *comm,
                 Stream *stream) const override;

  void ReduceScatter(const void *sendbuff, void *recvbuff,
                     size_t recv_count, DataType dtype,
                     nn::parallel::comm::ReduceOpType reduce_op,
                     const CclComm *comm, Stream *stream) const override;

  void AlltoAll(const void *sendbuff, void *recvbuff, size_t count,
                DataType dtype, const CclComm *comm,
                Stream *stream) const override;

  void Send(const void *buff, size_t count, DataType dtype, int peer,
            const CclComm *comm, Stream *stream) const override;

  void Recv(void *buff, size_t count, DataType dtype, int peer,
            const CclComm *comm, Stream *stream) const override;
};

}  // namespace infini_train::core
