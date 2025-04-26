// RUN: rv-opt %s -reg-alloc > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo(%arg0: !rvir.reg<11>, %arg1: !rvir.reg<12>) -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   %1 = "rvir.ADDI"(%0) <{imm = 23 : i32}> : (!rvir.reg<0>) -> !rvir.reg<4>
// CHECK:   %2 = "rvir.SLT"(%arg0, %1) : (!rvir.reg<11>, !rvir.reg<4>) -> !rvir.reg<4>
// CHECK:   %3 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   "rvir.BEQ"(%2, %3)[^bb2] : (!rvir.reg<4>, !rvir.reg<0>) -> ()
// CHECK: ^bb1(%4: !rvir.reg<10>):  // pred: ^bb2
// CHECK:   return %4 : !rvir.reg<10>
// CHECK: ^bb2:  // pred: ^bb0
// CHECK:   %5 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   "rvir.BEQ"(%arg1, %5)[^bb1] : (!rvir.reg<12>, !rvir.reg<0>) -> ()
// CHECK: }

module {
    func.func @foo(%arg0: !rvir.reg, %arg1: !rvir.reg) -> !rvir.reg {
      %0 = "rvir.Const"() : () -> !rvir.reg<0>
      %1 = "rvir.ADDI"(%0) <{imm = 23 : i32}> : (!rvir.reg<0>) -> !rvir.reg
      %2 = "rvir.SLT"(%arg0, %1) : (!rvir.reg, !rvir.reg) -> !rvir.reg
      %3 = "rvir.Const"() : () -> !rvir.reg<0>
      "rvir.BEQ"(%2, %3)[^bb2] : (!rvir.reg, !rvir.reg<0>) -> ()
    ^bb1(%4: !rvir.reg):  // pred: ^bb2
      return %4 : !rvir.reg
    ^bb2:  // pred: ^bb0
      %5 = "rvir.Const"() : () -> !rvir.reg<0>
      "rvir.BEQ"(%arg1, %5)[^bb1] : (!rvir.reg, !rvir.reg<0>) -> ()
    }
}
