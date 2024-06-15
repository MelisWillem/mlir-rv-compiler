add_mlir_dialect(RVIROps rvir)
add_mlir_doc(RVIROps RVIROps RVIR/ -gen-op-doc)
add_mlir_doc(RVIRDialect RVIRDialect RVIR/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS RVIRPasses.td)
mlir_tablegen(RVIRPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRRVIRPassesIncGen)
