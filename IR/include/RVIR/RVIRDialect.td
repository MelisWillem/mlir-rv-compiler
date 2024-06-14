#ifndef RVIR_DIALECT
#define RVIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RVIR dialect definition.
//===----------------------------------------------------------------------===//

def RVIR_Dialect : Dialect {
    let name = "rvir";
    let summary = "a dialect of riscv instruction set";
    let description = [{
    }];
    let cppNamespace = "::mlir::rvir";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base RVIR operation definition.
//===----------------------------------------------------------------------===//

class RVIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<RVIR_Dialect, mnemonic, traits>;

#endif // RVIR_DIALECT
