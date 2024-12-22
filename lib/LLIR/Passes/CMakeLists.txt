add_mlir_dialect_library(LLIRPasses
 HLIRToLLIRPass.cpp

 DEPENDS

 LINK_LIBS PUBLIC
 MLIRIR
 MLIRPass
 # LLIROps
 # RHLIROps
 )
