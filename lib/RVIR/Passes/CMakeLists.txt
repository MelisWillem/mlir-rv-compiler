add_mlir_dialect_library(RVIRPasses
 ToRVPass.cpp

 DEPENDS

 LINK_LIBS PUBLIC
 MLIRIR
 MLIRPass
 # LLIROps
 # RHLIROps
 )
