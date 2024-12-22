// RUN: rv-opt %s -HLIR-LLIR > %t.mlir
module {
  func.func @foo(%arg0: i32, %arg1: i1) -> i32 {
    %0 = "hlir.alloca"() <{allocaType = i32, name = "number"}> : () -> !hlir.ptr
    "hlir.Store"(%arg0, %0) : (i32, !hlir.ptr) -> ()
    %1 = "hlir.alloca"() <{allocaType = i1, name = "flag"}> : () -> !hlir.ptr
    "hlir.Store"(%arg1, %1) : (i1, !hlir.ptr) -> ()
    %2 = "hlir.Load"(%0) : (!hlir.ptr) -> i32
    %3 = "hlir.Constant"() <{constant = 23 : i32}> : () -> i32
    %4 = "hlir.Compare"(%2, %3) <{type = #hlir<CmpType greather>}> : (i32, i32) -> i1
    "hlir.If"(%4)[^bb1, ^bb2] : (i1) -> ()
  ^bb1:  // pred: ^bb0
    %5 = "hlir.Load"(%0) : (!hlir.ptr) -> i32
    "hlir.Return"(%5) : (i32) -> ()
  ^bb2:  // pred: ^bb0
    %6 = "hlir.Load"(%1) : (!hlir.ptr) -> i1
    "hlir.If"(%6)[^bb3, ^bb4] : (i1) -> ()
  ^bb3:  // pred: ^bb2
    %7 = "hlir.Constant"() <{constant = 1 : i32}> : () -> i32
    "hlir.Return"(%7) : (i32) -> ()
  ^bb4:  // pred: ^bb2
    %8 = "hlir.Constant"() <{constant = -1 : i32}> : () -> i32
    "hlir.Return"(%8) : (i32) -> ()
  }
}
