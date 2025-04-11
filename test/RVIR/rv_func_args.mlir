// RUN: rv-opt %s -to-rv -remove-dead-values  > %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// CHECK: func.func @only_arg(%arg0: !rvir.reg<None>) {
// CHECK:   return
// CHECK: }
// CHECK: func.func @arg_with_return(%arg0: !rvir.reg<None>) -> !rvir.reg<None> {
// CHECK:   return %arg0 : !rvir.reg<None>
// CHECK: }
// CHECK: func.func @arg_with_more_then_one_arg_and_return(%arg0: !rvir.reg<None>, %arg1: !rvir.reg<None>, %arg2: !rvir.reg<None>) -> !rvir.reg<None> {
// CHECK:   return %arg0 : !rvir.reg<None>
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