#!/bin/bash
clang-18 --target=riscv32 -march=rv32i test.cpp -O3  -S -o -
