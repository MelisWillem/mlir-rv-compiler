#ifndef RVIR_TYPES
#define RVIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "RVIR/RVIRDialect.td"

//===----------------------------------------------------------------------===//
// RVIR type definitions
//===----------------------------------------------------------------------===//

class RVIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RVIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Need to add some type or the default parsers won't work.
// Once there is one custom type in the dialect this can be removed.
def RVIR_CustomType : RVIR_Type<"Dummy", "dummy"> {}

#endif // RVIR_TYPES
