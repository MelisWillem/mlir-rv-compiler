#ifndef LLIR_PASS
#define LLIR_PASS

include "mlir/Pass/PassBase.td"

def HLToLLIRPass : Pass<"HLIR-LLIR", "::mlir::ModuleOp"> {
    let dependentDialects = [
        "mlir::hlir::HLIRDialect",
        "mlir::llir::LLIRDialect",
        "mlir::arith::ArithDialect", 
        "mlir::func::FuncDialect", 
        "mlir::cf::ControlFlowDialect"
         ];
}

#endif // LLIR_PASS
