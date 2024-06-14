#ifndef LLIR_DIALECT
#define LLIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLIR dialect definition.
//===----------------------------------------------------------------------===//

def LLIR_Dialect : Dialect {
    let name = "llir";
    let summary = "low lever ir";
    let description = [{
    }];
    let cppNamespace = "::mlir::llir";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base LLIR operation definition.
//===----------------------------------------------------------------------===//

class LLIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<LLIR_Dialect, mnemonic, traits>;

#endif // LLIR_DIALECT
