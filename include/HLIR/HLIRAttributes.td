#ifndef HLIR_ATTRIBUTES_H
#define HLIR_ATTRIBUTES_H

include "HLIRDialect.td"
include "mlir/IR/EnumAttr.td"
include "HLIREnums.td"

def CmpTypeAttr : EnumAttr<HLIR_Dialect, HLIRComparisonType, "CmpType">;

#endif // HLIR_ATTRIBUTES_H