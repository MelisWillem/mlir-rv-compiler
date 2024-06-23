#ifndef HLIR_DIALECT
#define HLIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// HLIR dialect definition.
//===----------------------------------------------------------------------===//

def HLIR_Dialect : Dialect {
    let name = "hlir";
    let summary = "high lever ir";
    let description = [{
    }];
    let cppNamespace = "::mlir::hlir";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base HLIR operation definition.
//===----------------------------------------------------------------------===//

class HLIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<HLIR_Dialect, mnemonic, traits>;

#endif // HLIR_DIALECT
