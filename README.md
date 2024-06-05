# mlir-experiment

Compile with command:

```
cmake ../ \
    -G Ninja \
    -DMLIR_DIR=${MLIR_BULD_PATH}/lib/cmake/mlir \
    -DLLVM_DIR_DIR={LLVM_BUILD_PATH}/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=${LIT_PATH}/lit
```
