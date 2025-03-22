#ifndef RVIR_RVIRPASSES_H
#define RVIR_RVIRPASSES_H

#include "RVIR/RVIRDialect.h"
#include "RVIR/RVIROps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace rvir {
#define GEN_PASS_DECL_TORV
#include "RVIR/RVIRPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RVIR/RVIRPasses.h.inc"
} // namespace rvir
} // namespace mlir

#endif
