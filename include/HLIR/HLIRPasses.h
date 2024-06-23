#ifndef HLIR_RVIRPASSES_H
#define HLIR_RVIRPASSES_H

#include "HLIR/HLIRDialect.h"
#include "HLIR/HLIROps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hlir {
#define GEN_PASS_DECL
#include "HLIR/HLIRPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "HLIR/HLIRPasses.h.inc"
} // namespace hlir
} // namespace mlir

#endif
