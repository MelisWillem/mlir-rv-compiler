// RUN: rv-opt %s -HLIR-LLIR > %t.mlir
module {
  func.func @foo(%arg0: i32, %arg1: i1) -> i32 {
    %alloca = memref.alloca() : memref<1xi32>
    %c0 = arith.constant 0 : index
    memref.store %arg0, %alloca[%c0] : memref<1xi32>
    %alloca_0 = memref.alloca() : memref<1xi1>
    %c0_1 = arith.constant 0 : index
    memref.store %arg1, %alloca_0[%c0_1] : memref<1xi1>
    %c0_2 = arith.constant 0 : index
    %0 = memref.load %alloca[%c0_2] : memref<1xi32>
    %c23_i32 = arith.constant 23 : i32
    %1 = arith.cmpi sgt, %0, %c23_i32 : i32
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %c0_3 = arith.constant 0 : index
    %2 = memref.load %alloca[%c0_3] : memref<1xi32>
    "hlir.Return"(%2) : (i32) -> ()
  ^bb2:  // pred: ^bb0
    %c0_4 = arith.constant 0 : index
    %3 = memref.load %alloca_0[%c0_4] : memref<1xi1>
    cf.cond_br %3, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %c1_i32 = arith.constant 1 : i32
    "hlir.Return"(%c1_i32) : (i32) -> ()
  ^bb4:  // pred: ^bb2
    %c-1_i32 = arith.constant -1 : i32
    "hlir.Return"(%c-1_i32) : (i32) -> ()
  }
}
