#include "LLIR/LLIRDialect.h"
#include "LLIR/LLIROps.h"
#include "LLIR/LLIRTypes.h"

using namespace mlir;
using namespace mlir::llir;

#include "LLIR/LLIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LLIR dialect.
//===----------------------------------------------------------------------===//

void LLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LLIR/LLIROps.cpp.inc"
      >();
  registerTypes();
}
