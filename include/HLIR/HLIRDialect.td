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
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];
}

#endif // HLIR_DIALECT
