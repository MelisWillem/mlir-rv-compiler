// RUN: rv-opt %s -mem2reg -to-rv > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo() -> !rvir.reg<None> {
// CHECK:   %0 = "rvir.ADDI"() <{imm = -1 : i32}> : () -> !rvir.reg<None>
// CHECK:   return %0 : !rvir.reg<None>
// CHECK: }

module {
  func.func @foo() -> i32 {
    %num = arith.constant -1 : i32
    return %num : i32
  }
}