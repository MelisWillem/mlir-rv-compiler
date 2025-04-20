#ifndef RVIR_PASS
#define RVIR_PASS

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ToRV : Pass<"to-rv", "func::FuncOp"> {
  let summary = "Translate functions to Riscv";
  let description = [{
    Translate functions to Riscv
  }];
  let dependentDialects = [
    "mlir::rvir::RVIRDialect"
  ];
  let options = RewritePassUtils.options;
}

#endif // RVIR_PASS
