#include "HLIR/HLIRDialect.h"
#include "HLIR/HLIROps.h"
#include "HLIR/HLIRTypes.h"

using namespace mlir;
using namespace mlir::hlir;

#include "HLIR/HLIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LLIR dialect.
//===----------------------------------------------------------------------===//

void HLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HLIR/HLIROps.cpp.inc"
      >();
  registerTypes();
  registerAttributes();
}
