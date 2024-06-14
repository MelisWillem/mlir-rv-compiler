#ifndef LLIR_TYPES
#define LLIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "LLIR/LLIRDialect.td"

//===----------------------------------------------------------------------===//
// LLIR type definitions
//===----------------------------------------------------------------------===//

class LLIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Need to add some type or the default parsers won't work.
// Once there is one custom type in the dialect this can be removed.
def RVIR_CustomType : LLIR_Type<"Dummy", "dummy"> {}

#endif // LLIR_TYPES
