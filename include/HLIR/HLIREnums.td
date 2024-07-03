#ifndef HLIR_ENUMS
#define HLIR_ENUMS

include "HLIRDialect.td"
include "mlir/IR/EnumAttr.td"

def HLIRComparisonType : I32EnumAttr<"CmpType",
    "Type of comparison",
    [
      I32EnumAttrCase<"smaller", 0>,
      I32EnumAttrCase<"greather", 1>,
      I32EnumAttrCase<"equal", 2>,
      I32EnumAttrCase<"notequal", 3>,
      I32EnumAttrCase<"eqsmaller", 4>,
      I32EnumAttrCase<"eqgreather", 5>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hlir";
}

#endif // HLIR_ENUMS