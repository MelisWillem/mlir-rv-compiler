// RUN: rv-opt %s -mem2reg -to-rv > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo() -> !rvir.reg {
// CHECK:   %0 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   %1 = "rvir.ADDI"(%0) <{imm = -1 : i32}> : (!rvir.reg<0>) -> !rvir.reg
// CHECK:   return %1 : !rvir.reg
// CHECK: }

module {
  func.func @foo() -> i32 {
    %num = arith.constant -1 : i32
    return %num : i32
  }
}