// RUN: rv-opt %s -mem2reg -to-rv > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: module {
// CHECK:   func.func @foo(%arg0: !rvir.reg, %arg1: !rvir.reg) -> !rvir.reg {
// CHECK:     %0 = "rvir.ADDI"() <{imm = 23 : i32}> : () -> !rvir.reg
// CHECK:     %1 = "rvir.SLT"(%arg0, %0) : (!rvir.reg, !rvir.reg) -> !rvir.reg
// CHECK:     %2 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:     "rvir.BEQ"(%1, %2)[^bb2] : (!rvir.reg, !rvir.reg<0>) -> ()
// CHECK:   ^bb1(%3: !rvir.reg):  // pred: ^bb2
// CHECK:     return %3 : !rvir.reg
// CHECK:   ^bb2:  // pred: ^bb0
// CHECK:     %4 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:     "rvir.BEQ"(%arg1, %4)[^bb1] : (!rvir.reg, !rvir.reg<0>) -> ()
// CHECK:   }
// CHECK: }

module {
  func.func @foo(%arg0: i32, %arg1: i1) -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c23_i32 = arith.constant 23 : i32
    %0 = arith.cmpi sgt, %arg0, %c23_i32 : i32
    cf.cond_br %0, ^bb1(%arg0 : i32), ^bb2
  ^bb1(%1: i32):  // 3 preds: ^bb0, ^bb2, ^bb2
    return %1 : i32
  ^bb2:  // pred: ^bb0
    cf.cond_br %arg1, ^bb1(%c1_i32 : i32), ^bb1(%c-1_i32 : i32)
  }
}