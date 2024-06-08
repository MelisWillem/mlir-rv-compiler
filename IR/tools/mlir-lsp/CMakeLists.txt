# To build mlir-lsp enable "-DLLVM_BUILD_TOOLS=on" in cmake
add_llvm_tool(mlir-lsp mlir-lsp.cpp)

target_link_libraries(mlir-lsp
    PRIVATE
    MLIRLspServerLib
    MLIRIR  
    MLIRStandalone
    MLIRArithDialect
)