#include "RVIR/RVIRDialect.h"
#include "RVIR/RVIROps.h"
#include "RVIR/RVIRTypes.h"

using namespace mlir;
using namespace mlir::rvir;

#include "RVIR/RVIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RVIR dialect.
//===----------------------------------------------------------------------===//

void RVIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RVIR/RVIROps.cpp.inc"
      >();
  registerTypes();
}
