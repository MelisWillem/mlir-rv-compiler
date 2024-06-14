#include "LLIR/LLIRTypes.h"

#include "LLIR/LLIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir::llir;

#define GET_TYPEDEF_CLASSES
#include "LLIR/LLIROpsTypes.cpp.inc"

void LLIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "LLIR/LLIROpsTypes.cpp.inc"
      >();
}
