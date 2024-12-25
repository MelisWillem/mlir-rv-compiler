// RUN: rv-opt %s -HLIR-LLIR > %t.mlir
module {
  func.func @foo(%arg0: i32, %arg1: i1) -> i32 {
    %0 = "hlir.alloca"() <{allocaType = i32, name = "number"}> : () -> !hlir.ptr
    "hlir.Store"(%arg0, %0) : (i32, !hlir.ptr) -> ()
    %1 = "hlir.alloca"() <{allocaType = i1, name = "flag"}> : () -> !hlir.ptr
    "hlir.Store"(%arg1, %1) : (i1, !hlir.ptr) -> ()
    %2 = "hlir.Load"(%0) : (!hlir.ptr) -> i32
    %c23_i32 = arith.constant 23 : i32
    %3 = arith.cmpi sgt, %2, %c23_i32 : i32
    cf.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %4 = "hlir.Load"(%0) : (!hlir.ptr) -> i32
    "hlir.Return"(%4) : (i32) -> ()
  ^bb2:  // pred: ^bb0
    %5 = "hlir.Load"(%1) : (!hlir.ptr) -> i1
    cf.cond_br %5, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %c1_i32 = arith.constant 1 : i32
    "hlir.Return"(%c1_i32) : (i32) -> ()
  ^bb4:  // pred: ^bb2
    %c-1_i32 = arith.constant -1 : i32
    "hlir.Return"(%c-1_i32) : (i32) -> ()
  }
}
