// RUN: rv-opt %s -mem2reg -to-rv > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: module {
// CHECK:   func.func @foo(%arg0: !rvir.reg<11>, %arg1: !rvir.reg<12>) -> !rvir.reg<10> {
// CHECK:     %0 = "rvir.ADDI"() <{imm = 23 : i32}> : () -> !rvir.reg<None>
// CHECK:     %1 = "rvir.SLT"(%arg0, %0) : (!rvir.reg<11>, !rvir.reg<None>) -> !rvir.reg<None>
// CHECK:     %2 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:     "rvir.BEQ"(%1, %2)[^bb2] : (!rvir.reg<None>, !rvir.reg<0>) -> ()
// CHECK:   ^bb1(%3: i32):  // pred: ^bb2
// CHECK:     %4 = "rvir.ValReg"(%3) : (i32) -> !rvir.reg<10>
// CHECK:     return %4 : !rvir.reg<10>
// CHECK:   ^bb2:  // pred: ^bb0
// CHECK:     %5 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:     "rvir.BEQ"(%arg1, %5)[^bb1] : (!rvir.reg<12>, !rvir.reg<0>) -> ()
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