#ifndef RVIR_TYPES
#define RVIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "RVIR/RVIRDialect.td"
include "mlir/IR/BuiltinAttributes.td"

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

def RegisterType : RVIR_Type<"Register", "reg"> {
  let summary = "Register type";
  let description = [{
    The `register` type is used to represent a register in the RISC-V dialect.
  }];

  let parameters = (ins DefaultValuedParameter<"std::optional<int>", "std::nullopt">:$id);
  let assemblyFormat = [{
    (`<` $id^ `>`)?
  }];
}


#endif // RVIR_TYPES
