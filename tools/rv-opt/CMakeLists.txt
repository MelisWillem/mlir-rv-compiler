get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIROptLib
        RVIR
        LLIR
        )
add_llvm_executable(rv-opt rv-opt.cpp)

llvm_update_compile_flags(rv-opt)
target_link_libraries(rv-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(rv-opt)
