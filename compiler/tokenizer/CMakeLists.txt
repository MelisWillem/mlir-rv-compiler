add_library(TokenLib
    ${CMAKE_CURRENT_LIST_DIR}/lib/tokenizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lib/tokens.cpp
    )
target_include_directories(TokenLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

add_executable(tokenizer ${CMAKE_CURRENT_LIST_DIR}/bin/convertToTokens.cpp)
target_link_libraries(tokenizer PUBLIC TokenLib)