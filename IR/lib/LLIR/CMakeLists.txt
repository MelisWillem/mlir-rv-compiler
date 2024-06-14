add_mlir_dialect_library(LLIR
        LLIRTypes.cpp
        LLIRDialect.cpp
        LLIROps.cpp
        LLIRPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/LLIR

        DEPENDS
        MLIRLLIROpsIncGen
        MLIRLLIRPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
