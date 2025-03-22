add_mlir_dialect_library(RVIR
        RVIRTypes.cpp
        RVIRDialect.cpp
        RVIROps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RVIR

        DEPENDS
        MLIRRVIROpsIncGen
        MLIRRVIRPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        RVIRPasses
      )

add_subdirectory(Passes)
