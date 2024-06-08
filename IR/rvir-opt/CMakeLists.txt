get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        RVIR
        )
add_llvm_executable(rvir-opt rvir-opt.cpp)

llvm_update_compile_flags(rvir-opt)
target_link_libraries(rvir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(rvir-opt)
