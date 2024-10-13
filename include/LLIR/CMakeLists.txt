include_directories("${CMAKE_CURRENT_LIST_DIR}/../HLIR")

add_mlir_dialect(LLIROps llir)
add_mlir_doc(LLIROps LLIROps LLIR/ -gen-op-doc)
add_mlir_doc(LLIRDialect LLIRDialect LLIR/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS LLIRPasses.td)
mlir_tablegen(LLIRPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRLLIRPassesIncGen)

# generate pdll files
add_mlir_pdll_library(MLIRHLIIRPDLLPatternsIncGen
  HLIRToLLIR.pdll
  HLIRToLLIR.inc.h
  )