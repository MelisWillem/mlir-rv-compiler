#include "HLIR/HLIRDialect.h"
#include"HLIR/HLIRAttributes.h"
#include "HLIR/HLIREnums.h"

#include"mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/OpImplementation.h" // needed for things like AsmParser
#include "mlir/IR/DialectImplementation.h" // needed for DialectAsmParser
#include "mlir/IR/Builders.h"


#define GET_ATTRDEF_CLASSES
#include"HLIR/HLIRAttributes.cpp.inc"


void mlir::hlir::HLIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include"HLIR/HLIRAttributes.cpp.inc"
      >();
}