// RUN: rv-opt %s -reg-alloc > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @foo() -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.Const"() : () -> !rvir.reg<0>
// CHECK:   %1 = "rvir.ADDI"(%0) <{imm = -1 : i32}> : (!rvir.reg<0>) -> !rvir.reg<4>
// CHECK:   %2 = "rvir.ADDI"(%0) <{imm = -2 : i32}> : (!rvir.reg<0>) -> !rvir.reg<5>
// CHECK:   %3 = "rvir.ADD"(%1, %2) : (!rvir.reg<4>, !rvir.reg<5>) -> !rvir.reg<10>
// CHECK:   %4 = "rvir.ADD"(%1, %3) : (!rvir.reg<4>, !rvir.reg<10>) -> !rvir.reg<4>
// CHECK:   return %3 : !rvir.reg<10>
// CHECK: }

module {
  func.func @foo() -> !rvir.reg {
    %nullreg = "rvir.Const"() : () -> !rvir.reg<0>
    %0 = "rvir.ADDI"(%nullreg) <{imm = -1 : i32}> : (!rvir.reg<0>) -> !rvir.reg
    %1 = "rvir.ADDI"(%nullreg) <{imm = -2 : i32}> : (!rvir.reg<0>) -> !rvir.reg
    %2 = "rvir.ADD"(%0, %1) : (!rvir.reg, !rvir.reg) -> !rvir.reg
    %3 = "rvir.ADD"(%0, %2) : (!rvir.reg, !rvir.reg) -> !rvir.reg
    return %2 : !rvir.reg
  }
}