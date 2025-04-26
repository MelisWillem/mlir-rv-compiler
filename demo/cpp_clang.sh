#!/bin/bash
clang-18 --target=riscv32 -march=rv32i cpp_clang_builld.cpp -O3  -S -o -
