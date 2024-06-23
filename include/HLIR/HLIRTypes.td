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

// Need to add some type or the default parsers won't work.
// Once there is one custom type in the dialect this can be removed.
def RVIR_CustomType : HLIR_Type<"Dummy", "dummy"> {}

#endif // HLIR_TYPES
