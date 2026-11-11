// 对应 InfiniTrain/src/core/ccl/cuda/nccl_impl.cc，把 NCCL 调用逐一换成 CNCL。
//
// CNCL 与 NCCL 的关键差异（已对 cncl.h 核对）：
//   1) 集合通信 API 多一个末参 cnrtQueue_t queue（对应 NCCL 的 cudaStream_t）。
//      框架传进来的 Stream* 经 dynamic_cast<MluStream*> 取 mlu_queue()。
//   2) UniqueId 叫 cnclCliqueId（128B + 8B hash），由 cnclGetCliqueId 产生。
//   3) 单进程多卡初始化用 cnclInitComms(comms, num_comm, dev_list, rank_list, nrank, clique_id)，
//      不是 NCCL 的 ncclCommInitAll。框架 CommInitAll(comms, ndev, devlist) 传进来的是
//      全局 rank 数组，这里把 rank_list[i]=devlist[i]、dev_list[i]=devlist[i]%DeviceCount()，
//      再 cnrtSetDevice 到每张卡后调 cnclInitComms。
//   4) cnclDataType_t：cnclHalf/cnclBfloat16/cnclFloat32/cnclInt8/cnclInt16/cnclInt32/
//      cnclUint8/cnclUint16/cnclUint32/cnclUint64（注意 cncl.h 无 cnclFloat64/cnclInt64）。
//   5) cnclReduceOp_t 只有 cnclSum/cnclProd/cnclMax/cnclMin，没有 cnclAvg。
//      框架 ReduceOpType::kAvg（DDP 默认 average_in_collective=true 时梯度 AllReduce、
//      以及 example 的 loss AllReduce 都会用到）在此实现为 cnclSum 后原地乘 1/nrank
//      （kernels::mlu::ScalarForwardLaunch，与集合通信同一条 queue，顺序天然保证）。
//      只映射 Sum 不缩放是错的：2 卡 smoke 的 loss 会变成 2 倍、梯度被放大 world_size 倍。
//   6) CNCL 有原生 cnclAlltoAll（见 cncl.h:1508），直接调用，无需 Send/Recv 回退。
//   7) CNCL 有 cnclGetCommAsyncError（见 cncl.h:1698），GetAsyncError 据此报告 CclStatus。

#include "cncl_impl.h"

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cnrt.h>
#include <glog/logging.h>

#include "infini_train/include/common/common.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

#include "cncl_common.h"
#include "common/common_mlu.h"
#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::core {
namespace {

// CNRT_CHECK / CNCL_CHECK 是在 common_mlu.h 里 #define 的预处理宏，不属于任何
// 命名空间，包含该头后即全局可用，无需 using。

inline const std::unordered_map<DataType, cnclDataType_t> &CnclDtypeMap() {
  static const std::unordered_map<DataType, cnclDataType_t> map = {
      {DataType::kINT8, cnclInt8},
      {DataType::kUINT8, cnclUint8},
      {DataType::kINT16, cnclInt16},
      {DataType::kUINT16, cnclUint16},
      {DataType::kINT32, cnclInt32},
      {DataType::kUINT32, cnclUint32},
      {DataType::kUINT64, cnclUint64},
      {DataType::kFLOAT16, cnclHalf},
      {DataType::kBFLOAT16, cnclBfloat16},
      {DataType::kFLOAT32, cnclFloat32},
  };
  return map;
}

// 框架 ReduceOpType → cnclReduceOp_t。kAvg 无对应算子，这里只返回 cnclSum；
// 平均语义由调用方在集合通信之后用 ScaleAvg 补上（见文件头注释 5）。
inline cnclReduceOp_t ToCnclReduceOp(nn::parallel::comm::ReduceOpType op) {
  switch (op) {
  case nn::parallel::comm::ReduceOpType::kSum:
    return cnclSum;
  case nn::parallel::comm::ReduceOpType::kProd:
    return cnclProd;
  case nn::parallel::comm::ReduceOpType::kMin:
    return cnclMin;
  case nn::parallel::comm::ReduceOpType::kMax:
    return cnclMax;
  case nn::parallel::comm::ReduceOpType::kAvg:
    return cnclSum;
  }
  LOG(FATAL) << "Unsupported ReduceOpType: "
             << static_cast<int>(op);
  return cnclSum;
}

inline cnclDataType_t ToCnclDtype(DataType dtype) {
  const auto &map = CnclDtypeMap();
  auto it = map.find(dtype);
  CHECK(it != map.end())
      << "CNCL does not support dtype " << static_cast<int>(dtype);
  return it->second;
}

inline cnclComm_t GetCnclComm(const CclComm *comm) {
  auto *cncl_comm = dynamic_cast<const CnclComm *>(comm);
  CHECK_NOTNULL(cncl_comm);
  return cncl_comm->cncl_comm();
}

inline void SetCnclComm(CclComm *comm, cnclComm_t cncl_comm) {
  auto *typed = dynamic_cast<CnclComm *>(comm);
  CHECK_NOTNULL(typed);
  typed->set_cncl_comm(cncl_comm);
}

inline const cnclCliqueId &GetCnclCliqueId(const CclUniqueId &unique_id) {
  auto *cncl_uid = dynamic_cast<const CnclUniqueId *>(&unique_id);
  CHECK_NOTNULL(cncl_uid);
  return *cncl_uid->cncl_clique_id();
}

inline cnrtQueue_t GetMluQueue(Stream *stream) {
  auto *mlu_stream = dynamic_cast<infini_train::core::mlu::MluStream *>(stream);
  CHECK_NOTNULL(mlu_stream);
  return mlu_stream->mlu_queue();
}

inline kernels::mlu::MluScalarType ToMluScalarType(DataType dtype) {
  switch (dtype) {
  case DataType::kFLOAT16:
    return kernels::mlu::MluScalarType::kF16;
  case DataType::kBFLOAT16:
    return kernels::mlu::MluScalarType::kBF16;
  case DataType::kFLOAT32:
    return kernels::mlu::MluScalarType::kF32;
  default:
    LOG(FATAL) << "CNCL kAvg scaling does not support dtype "
               << static_cast<int>(dtype);
  }
  return kernels::mlu::MluScalarType::kF32;
}

// kAvg = Sum + 原地乘 1/nrank。CNCL 无 cnclAvg，Sum 完成后用后端的标量乘
// kernel 在原地缩放：ScalarForwardLaunch 每元素读自己写自己，out==in 安全；
// 它跑在 process_group 传入的同一台设备当前 stream（MluStream::mlu_queue）
// 上，与前面的 cncl 集合通信天然保序，无需额外同步。
// root_only：Reduce 只有 root 收到有效结果，非 root 不缩放。
void ScaleAvg(void *buf, size_t count, DataType dtype, const CclComm *comm,
              Stream *stream, bool root_only, int root) {
  int nrank = 0;
  CNCL_CHECK(cnclGetCommCount(&nrank, GetCnclComm(comm)));
  CHECK_GT(nrank, 0);
  if (root_only) {
    int myrank = 0;
    CNCL_CHECK(cnclGetCommRank(&myrank, GetCnclComm(comm)));
    if (myrank != root) {
      return;
    }
  }
  kernels::mlu::ScalarForwardLaunch(buf, buf, 1.0f / nrank,
                                    kernels::mlu::MluBinaryOp::kMul,
                                    ToMluScalarType(dtype),
                                    static_cast<int64_t>(count),
                                    GetMluQueue(stream));
}

}  // namespace

Device::DeviceType CnclImpl::Type() const {
  return Device::DeviceType::kPrivateUse1;
}

void CnclImpl::GroupStart() const { CNCL_CHECK(cnclGroupStart()); }

void CnclImpl::GroupEnd() const { CNCL_CHECK(cnclGroupEnd()); }

void CnclImpl::GetAsyncError(const CclComm *comm,
                             CclStatus *async_error) const {
  // cnclGetCommAsyncError 返回 CNCL_RET_SUCCESS 表示无异步错误，否则返回错误码。
  // 这里只报告状态、不直接 FATAL：框架会在适当时机处理 kError。
  cnclResult_t cncl_err = cnclGetCommAsyncError(GetCnclComm(comm));
  if (async_error != nullptr) {
    *async_error = (cncl_err == CNCL_RET_SUCCESS) ? CclStatus::kSuccess
                                                  : CclStatus::kError;
  }
}

void CnclImpl::GetUniqueId(CclUniqueId **unique_id) const {
  CHECK_NOTNULL(unique_id);
  if (*unique_id == nullptr) {
    *unique_id = new CnclUniqueId();
  }
  auto *cncl_uid = dynamic_cast<CnclUniqueId *>(*unique_id);
  CHECK_NOTNULL(cncl_uid);
  CNCL_CHECK(cnclGetCliqueId(cncl_uid->cncl_clique_id()));
}

// 框架单进程多卡路径（process_group.cc:83 InitSingleProcess）调用本函数，
// devlist 传进来的是全局 rank 数组（device_indices = [0,1,...,ndev-1]）。
// CNCL 的 cnclInitComms 需要 dev_list（物理卡号）+ rank_list（逻辑 rank），
// 这里把 rank_list[i]=devlist[i]、dev_list[i]=devlist[i]%DeviceCount()，
// 先 cnrtSetDevice 到每张卡，再用同一个 clique_id 一次性初始化所有卡。
void CnclImpl::CommInitAll(CclComm **comms, int ndev,
                           const int *devlist) const {
  CHECK_NOTNULL(comms);
  CHECK_GT(ndev, 0);
  CHECK_NOTNULL(devlist);

  auto *guard_impl = GetDeviceGuardImpl(Device::DeviceType::kPrivateUse1);
  const int device_count = guard_impl->DeviceCount();
  CHECK_GT(device_count, 0);

  cnclCliqueId clique_id{};
  CNCL_CHECK(cnclGetCliqueId(&clique_id));

  std::vector<int> dev_list(static_cast<size_t>(ndev));
  std::vector<int> rank_list(static_cast<size_t>(ndev));
  std::vector<cnclComm_t> cncl_comms(static_cast<size_t>(ndev), nullptr);

  int current_device = -1;
  CNRT_CHECK(cnrtGetDevice(&current_device));

  for (int i = 0; i < ndev; ++i) {
    rank_list[static_cast<size_t>(i)] = devlist[i];
    dev_list[static_cast<size_t>(i)] = devlist[i] % device_count;
    if (comms[static_cast<size_t>(i)] == nullptr) {
      comms[static_cast<size_t>(i)] = new CnclComm();
    }
  }

  // cnclInitComms 签名：cnclComm_t* comms, int num_comm,
  // const int* dev_list, const int* rank_list, int nrank, cnclCliqueId_t clique_id。
  // 注意 cnclCliqueId_t 是 typedef struct cnclCliqueId*（指针），所以传 &clique_id。
  // 它内部会逐卡 cnrtSetDevice，不需要调用方预先切卡。
  CNCL_CHECK(cnclInitComms(cncl_comms.data(), ndev, dev_list.data(),
                           rank_list.data(), ndev, &clique_id));

  for (int i = 0; i < ndev; ++i) {
    SetCnclComm(comms[static_cast<size_t>(i)],
                cncl_comms[static_cast<size_t>(i)]);
  }

  // 恢复调用前的 device
  CNRT_CHECK(cnrtSetDevice(current_device));
}

void CnclImpl::CommInitRank(CclComm **comm, int nranks,
                            const CclUniqueId &unique_id, int rank) const {
  CHECK_NOTNULL(comm);
  CHECK_GT(nranks, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, nranks);

  if (*comm == nullptr) {
    *comm = new CnclComm();
  }

  // cnclCommInitRankConfig 签名：cnclComm_t*, int nrank,
  // const cnclCliqueId_t clique_id, int myrank, cnclCommConfig_t config。
  // clique_id 是 cnclCliqueId*（typedef struct cnclCliqueId*），需要取地址。
  // config 传 nullptr 表示用默认配置（cncl.h 注释明确支持）。
  auto *cncl_uid = dynamic_cast<const CnclUniqueId *>(&unique_id);
  CHECK_NOTNULL(cncl_uid);
  cnclCliqueId clique_id = *cncl_uid->cncl_clique_id(); // 拷一份，取地址传给 C 接口
  cnclComm_t cncl_comm = nullptr;
  cnclCommConfig_t config = nullptr;
  CNCL_CHECK(cnclCommInitRankConfig(&cncl_comm, nranks, &clique_id, rank, config));
  SetCnclComm(*comm, cncl_comm);
}

void CnclImpl::CommDestroy(CclComm *comm) const {
  if (comm == nullptr) {
    return;
  }
  CNCL_CHECK(cnclFreeComm(GetCnclComm(comm)));
  SetCnclComm(comm, nullptr);
}

void CnclImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count,
                         DataType dtype,
                         nn::parallel::comm::ReduceOpType reduce_op,
                         const CclComm *comm, Stream *stream) const {
  CNCL_CHECK(cnclAllReduce(sendbuff, recvbuff, count, ToCnclDtype(dtype),
                           ToCnclReduceOp(reduce_op), GetCnclComm(comm),
                           GetMluQueue(stream)));
  if (reduce_op == nn::parallel::comm::ReduceOpType::kAvg) {
    ScaleAvg(recvbuff, count, dtype, comm, stream, /*root_only=*/false,
             /*root=*/0);
  }
}

void CnclImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count,
                         DataType dtype, int root, const CclComm *comm,
                         Stream *stream) const {
  CNCL_CHECK(cnclBroadcast(sendbuff, recvbuff, count, ToCnclDtype(dtype), root,
                           GetCnclComm(comm), GetMluQueue(stream)));
}

void CnclImpl::Reduce(const void *sendbuff, void *recvbuff, size_t count,
                      DataType dtype,
                      nn::parallel::comm::ReduceOpType reduce_op, int root,
                      const CclComm *comm, Stream *stream) const {
  CNCL_CHECK(cnclReduce(sendbuff, recvbuff, count, ToCnclDtype(dtype),
                        ToCnclReduceOp(reduce_op), root, GetCnclComm(comm),
                        GetMluQueue(stream)));
  if (reduce_op == nn::parallel::comm::ReduceOpType::kAvg) {
    ScaleAvg(recvbuff, count, dtype, comm, stream, /*root_only=*/true, root);
  }
}

void CnclImpl::AllGather(const void *sendbuff, void *recvbuff, size_t count,
                         DataType dtype, const CclComm *comm,
                         Stream *stream) const {
  CNCL_CHECK(cnclAllGather(sendbuff, recvbuff, count, ToCnclDtype(dtype),
                           GetCnclComm(comm), GetMluQueue(stream)));
}

void CnclImpl::ReduceScatter(const void *sendbuff, void *recvbuff,
                             size_t recv_count, DataType dtype,
                             nn::parallel::comm::ReduceOpType reduce_op,
                             const CclComm *comm, Stream *stream) const {
  CNCL_CHECK(cnclReduceScatter(sendbuff, recvbuff, recv_count,
                               ToCnclDtype(dtype), ToCnclReduceOp(reduce_op),
                               GetCnclComm(comm), GetMluQueue(stream)));
  if (reduce_op == nn::parallel::comm::ReduceOpType::kAvg) {
    // scatter 后每个 rank 拿到的是自己的 chunk，各自缩放即可。
    ScaleAvg(recvbuff, recv_count, dtype, comm, stream, /*root_only=*/false,
             /*root=*/0);
  }
}

// CNCL 有原生 cnclAlltoAll（见 cncl.h:1508），直接调用，不走 Send/Recv 回退。
void CnclImpl::AlltoAll(const void *sendbuff, void *recvbuff, size_t count,
                        DataType dtype, const CclComm *comm,
                        Stream *stream) const {
  CHECK_NE(sendbuff, recvbuff) << "CnclImpl::AlltoAll does not support in-place.";
  CNCL_CHECK(cnclAlltoAll(sendbuff, recvbuff, count, ToCnclDtype(dtype),
                          GetCnclComm(comm), GetMluQueue(stream)));
}

void CnclImpl::Send(const void *buff, size_t count, DataType dtype, int peer,
                    const CclComm *comm, Stream *stream) const {
  CNCL_CHECK(cnclSend(const_cast<void *>(buff), count, ToCnclDtype(dtype), peer,
                      GetCnclComm(comm), GetMluQueue(stream)));
}

void CnclImpl::Recv(void *buff, size_t count, DataType dtype, int peer,
                    const CclComm *comm, Stream *stream) const {
  CNCL_CHECK(cnclRecv(buff, count, ToCnclDtype(dtype), peer,
                      GetCnclComm(comm), GetMluQueue(stream)));
}

INFINI_TRAIN_REGISTER_CCL_IMPL(Device::DeviceType::kPrivateUse1, CnclImpl)

}  // namespace infini_train::core
