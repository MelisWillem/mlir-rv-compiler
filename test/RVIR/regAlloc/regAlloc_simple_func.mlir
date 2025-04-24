// RUN: rv-opt %s -reg-alloc > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo(%arg0: !rvir.reg<11>, %arg1: !rvir.reg<12>) -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.ADDI"() <{imm = 23 : i32}> : () -> !rvir.reg<4>
// CHECK:   %1 = "rvir.SLT"(%arg0, %0) : (!rvir.reg<11>, !rvir.reg<4>) -> !rvir.reg<4>
// CHECK:   %2 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   "rvir.BEQ"(%1, %2)[^bb2] : (!rvir.reg<4>, !rvir.reg<0>) -> ()
// CHECK: ^bb1(%3: !rvir.reg<10>):  // pred: ^bb2
// CHECK:   return %3 : !rvir.reg<10>
// CHECK: ^bb2:  // pred: ^bb0
// CHECK:   %4 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   "rvir.BEQ"(%arg1, %4)[^bb1] : (!rvir.reg<12>, !rvir.reg<0>) -> ()
// CHECK: }

module {
    func.func @foo(%arg0: !rvir.reg, %arg1: !rvir.reg) -> !rvir.reg {
      %0 = "rvir.ADDI"() <{imm = 23 : i32}> : () -> !rvir.reg
      %1 = "rvir.SLT"(%arg0, %0) : (!rvir.reg, !rvir.reg) -> !rvir.reg
      %2 = "rvir.Const"() : () -> !rvir.reg<0>
      "rvir.BEQ"(%1, %2)[^bb2] : (!rvir.reg, !rvir.reg<0>) -> ()
    ^bb1(%3: !rvir.reg):  // pred: ^bb2
      return %3 : !rvir.reg
    ^bb2:  // pred: ^bb0
      %4 = "rvir.Const"() : () -> !rvir.reg<0>
      "rvir.BEQ"(%arg1, %4)[^bb1] : (!rvir.reg, !rvir.reg<0>) -> ()
    }
}
