//===- RVIRDialect.td - RVIR dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
