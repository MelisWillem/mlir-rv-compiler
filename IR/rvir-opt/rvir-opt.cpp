//===- RVIR-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "RVIR/RVIRDialect.h"
#include "RVIR/RVIRPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::rvir::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::rvir::RVIRDialect,
                  mlir::arith::ArithDialect, 
                  mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "RVIR optimizer driver\n", registry));
}
