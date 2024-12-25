add_mlir_dialect_library(HLIR
        HLIRTypes.cpp
        HLIRDialect.cpp
        HLIROps.cpp
        HLIREnums.cpp
        HLIRAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/HLIR

        DEPENDS
        MLIRIR
        MLIRHLIROpsIncGen
        MLIRHLIREnumsIncGen
        MLIRHLIRAttributesIncGen
        MLIRHLIRPassesIncGen

        LINK_LIBS PUBLIC
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRSupport
        MLIRMemRefDialect
        )
