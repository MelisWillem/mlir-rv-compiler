add_mlir_dialect(HLIROps hlir)
add_mlir_doc(HLIROps HLIROps HLIR/ -gen-op-doc)
add_mlir_doc(HLIRDialect HLIRDialect HLIR/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS HLIRPasses.td)
mlir_tablegen(HLIRPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRHLIRPassesIncGen)
