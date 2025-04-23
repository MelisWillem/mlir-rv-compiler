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
    "mlir::rvir::RVIRDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect"
  ];
  let options = RewritePassUtils.options;
}

def RegAlloc : Pass<"reg-alloc", "func::FuncOp"> {
  let summary = "register allocation";
  let description = [{
    Register allocation for Riscv functions.
  }];
  let dependentDialects = [
    "mlir::rvir::RVIRDialect",
  ];
  let options = RewritePassUtils.options;
}

#endif // RVIR_PASS
