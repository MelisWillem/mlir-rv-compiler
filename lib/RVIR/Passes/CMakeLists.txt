add_mlir_dialect_library(RVIRPasses
 ToRVPass.cpp
 RegAllocPass.cpp

 DEPENDS

 LINK_LIBS PUBLIC
 MLIRIR
 MLIRPass
 # LLIROps
 # RHLIROps
 )
