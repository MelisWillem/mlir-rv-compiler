#ifndef LLIR_TYPES
#define LLIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "HLIR/HLIRDialect.td"

//===----------------------------------------------------------------------===//
// HLIR type definitions
//===----------------------------------------------------------------------===//
class HLIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HLIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def HLIR_PtrType : HLIR_Type<"Pointer", "ptr"> {}

#endif // HLIR_TYPES
