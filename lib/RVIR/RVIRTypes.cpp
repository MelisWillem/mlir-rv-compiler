#include "RVIR/RVIRTypes.h"

#include "RVIR/RVIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir::rvir;

#define GET_TYPEDEF_CLASSES
#include "RVIR/RVIROpsTypes.cpp.inc"

void RVIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RVIR/RVIROpsTypes.cpp.inc"
      >();
}
