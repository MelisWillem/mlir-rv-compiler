add_mlir_dialect(HLIROps hlir)
add_mlir_doc(HLIROps HLIROps HLIR/ -gen-op-doc)
add_mlir_doc(HLIRDialect HLIRDialect HLIR/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS HLIRPasses.td)
mlir_tablegen(HLIRPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRHLIRPassesIncGen)

set(LLVM_TARGET_DEFINITIONS HLIREnums.td)
mlir_tablegen(HLIREnums.h.inc -gen-enum-decls)
mlir_tablegen(HLIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHLIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS HLIRAttributes.td)
mlir_tablegen(HLIRAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=hlir)
mlir_tablegen(HLIRAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hlir)
add_public_tablegen_target(MLIRHLIRAttributesIncGen)
