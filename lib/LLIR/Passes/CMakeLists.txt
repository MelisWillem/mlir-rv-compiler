add_mlir_dialect_library(LLIRPasses
 HLIRToLLIRPass.cpp

 DEPENDS
 MLIRHLIIRPDLLPatternsIncGen

 LINK_LIBS PUBLIC
 MLIRIR
 MLIRPass
 # LLIROps
 # RHLIROps
 )