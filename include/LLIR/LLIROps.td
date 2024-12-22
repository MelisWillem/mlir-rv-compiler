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

def ConstantOp : LLIR_Op<"Constant">{
  let arguments = (ins AnyAttrOf<[I1Attr, I32Attr]>:$constant);

  let results = (outs AnyTypeOf<[I32, I1]>:$value);
}

#endif // LLIR_OPS