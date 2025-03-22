// RUN: rv-opt %s -to-rv -remove-dead-values  > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @only_arg(%arg0: !rvir.reg<10>) {
// CHECK:   return
// CHECK: }
// CHECK: func.func @arg_with_return(%arg0: !rvir.reg<11>) -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.ADDI"(%arg0) <{imm = 0 : i32}> : (!rvir.reg<11>) -> !rvir.reg<10>
// CHECK:   return %0 : !rvir.reg<10>
// CHECK: }
// CHECK: func.func @arg_with_more_then_one_arg_and_return(%arg0: !rvir.reg<11>, %arg1: !rvir.reg<12>, %arg2: !rvir.reg<13>) -> !rvir.reg<10> {
// CHECK:   %0 = "rvir.ADDI"(%arg0) <{imm = 0 : i32}> : (!rvir.reg<11>) -> !rvir.reg<10>
// CHECK:   return %0 : !rvir.reg<10>
// CHECK: }

module {
  func.func @only_arg(%arg0: i32) {
    return
  }

  func.func @arg_with_return(%arg0: i32) -> i32 {
    return %arg0 : i32
  }

  func.func @arg_with_more_then_one_arg_and_return(
    %arg0: i32,
    %arg1: i32,
    %arg2: i32) -> i32 {
    return %arg0 : i32
  }
}