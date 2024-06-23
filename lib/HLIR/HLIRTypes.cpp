#include "HLIR/HLIRTypes.h"

#include "HLIR/HLIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir::hlir;

#define GET_TYPEDEF_CLASSES
#include "HLIR/HLIROpsTypes.cpp.inc"

void HLIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "HLIR/HLIROpsTypes.cpp.inc"
      >();
}
