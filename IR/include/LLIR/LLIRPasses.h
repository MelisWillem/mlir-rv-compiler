#ifndef LLIR_RVIRPASSES_H
#define LLIR_RVIRPASSES_H

#include "LLIR/LLIRDialect.h"
#include "LLIR/LLIROps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llir {
#define GEN_PASS_DECL
#include "LLIR/LLIRPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "LLIR/LLIRPasses.h.inc"
} // namespace llir
} // namespace mlir

#endif
