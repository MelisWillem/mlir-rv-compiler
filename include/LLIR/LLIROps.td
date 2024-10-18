#ifndef LLIR_OPS
#define LLIR_OPS

include "LLIR/LLIRTypes.td"
include "mlir/IR/CommonTypeConstraints.td"

// The control flow dialect already contains a conditional branch
// def BrIf : LLIR_Op<"BrIf">{
//     let description = "branch if a condition is true";
// 
//     // todo::how to represent a jump?
//     let arguments = (ins I1:$cond);
// }

#endif // LLIR_OPS