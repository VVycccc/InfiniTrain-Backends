#include "cncl_common.h"

#include <cstring>

#include "glog/logging.h"

namespace infini_train::core {

CnclComm::CnclComm() = default;

CnclComm::CnclComm(cnclComm_t comm) : cncl_comm_(comm) {}

cnclComm_t CnclComm::cncl_comm() const { return cncl_comm_; }

void CnclComm::set_cncl_comm(cnclComm_t comm) { cncl_comm_ = comm; }

CnclUniqueId::CnclUniqueId() = default;

CnclUniqueId::CnclUniqueId(const cnclCliqueId &id) : id_(id) {}

size_t CnclUniqueId::Size() const { return sizeof(id_); }

const void *CnclUniqueId::Data() const { return &id_; }

void CnclUniqueId::Load(const void *src, size_t size) {
  CHECK_NOTNULL(src);
  CHECK_EQ(size, sizeof(id_))
      << "CnclUniqueId size mismatch: expected " << sizeof(id_) << ", got " << size;
  std::memcpy(&id_, src, sizeof(id_));
}

cnclCliqueId *CnclUniqueId::cncl_clique_id() { return &id_; }

const cnclCliqueId *CnclUniqueId::cncl_clique_id() const { return &id_; }

}  // namespace infini_train::core
