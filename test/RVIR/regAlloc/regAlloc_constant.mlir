// RUN: rv-opt %s -reg-alloc > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo() -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.ADDI"() <{imm = -1 : i32}> : () -> !rvir.reg<10>
// CHECK:   return %0 : !rvir.reg<10>
// CHECK: }

module {
  func.func @foo() -> !rvir.reg {
    %0 = "rvir.ADDI"() <{imm = -1 : i32}> : () -> !rvir.reg
    return %0 : !rvir.reg
  }
}