// RUN: rvir-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = rvir.foo %{{.*}} : i32
        // %res = rvir.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @mlir_types(%arg0: !rvir.custom<"10">)
    func.func @mlir_types(%arg0: !rvir.custom<"10">) {
        return
    }
}
