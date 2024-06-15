# The llvm does not make use of rtti,
# let's disable it as well.
add_definitions("-fno-rtti")

add_executable(cspl main.cpp)
target_link_libraries(cspl PRIVATE LLVMSupport)