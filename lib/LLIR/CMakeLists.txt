add_mlir_dialect_library(LLIR
        LLIRTypes.cpp
        LLIRDialect.cpp
        LLIROps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/LLIR

        DEPENDS
        MLIRLLIROpsIncGen
        MLIRLLIRPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRControlFlowDialect
        LLIRPasses
        )

add_subdirectory(Passes)
