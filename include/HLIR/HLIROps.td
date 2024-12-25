#ifndef LIR_OPS
#define LIR_OPS

include "HLIR/HLIRTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "HLIREnums.td"
include "HLIRAttributes.td"

//===----------------------------------------------------------------------===//
// Base HLIR operation definition.
//===----------------------------------------------------------------------===//

class HLIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<HLIR_Dialect, mnemonic, traits>;

#endif // HLIR_OPS
