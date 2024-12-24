add_library(ParserLib
    ${CMAKE_CURRENT_LIST_DIR}/lib/parser.cpp
    )
target_include_directories(ParserLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(
  ParserLib 

  PRIVATE
  TokenLib
  MLIRIR
  MLIRSupport
  HLIR
  MLIRControlFlowDialect)

# add_executable(tokenizer ${CMAKE_CURRENT_LIST_DIR}/bin/convertToTokens.cpp)
# target_link_libraries(tokenizer PUBLIC TokenLib)
