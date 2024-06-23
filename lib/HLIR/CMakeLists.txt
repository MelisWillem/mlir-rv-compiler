add_mlir_dialect_library(HLIR
        HLIRTypes.cpp
        HLIRDialect.cpp
        HLIROps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/HLIR

        DEPENDS
        MLIRHLIROpsIncGen
        MLIRHLIRPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
