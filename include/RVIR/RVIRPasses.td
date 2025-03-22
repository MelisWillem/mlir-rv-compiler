#ifndef RVIR_PASS
#define RVIR_PASS

include "mlir/Pass/PassBase.td"

def ToRV : Pass<"to-rv", "func::FuncOp"> {
  let summary = "Translate functions to Riscv";
  let description = [{
    Translate functions to Riscv
  }];
  let dependentDialects = [
    "mlir::rvir::RVIRDialect"
  ];
}

#endif // RVIR_PASS
