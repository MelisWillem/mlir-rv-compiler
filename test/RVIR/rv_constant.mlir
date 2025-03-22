// RUN: rv-opt %s -mem2reg -to-rv > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo() -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.ADDI"() <{imm = -1 : i32}> : () -> !rvir.reg<None>
// CHECK:   %1 = "rvir.ADDI"(%0) <{imm = 0 : i32}> : (!rvir.reg<None>) -> !rvir.reg<10>
// CHECK:   return %1 : !rvir.reg<10>
// CHECK: }

module {
  func.func @foo() -> i32 {
    %num = arith.constant -1 : i32
    return %num : i32
  }
}