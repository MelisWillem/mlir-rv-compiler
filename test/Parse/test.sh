# RUN: tokenizer %S/simple_func.spl > %t.tokens
# RUN: diff %t.tokens %S/simple_func.tokens
# RUN: cspl -i %S/simple_func.spl -o %t.mlir -HIL
# RUN: diff %t.mlir %S/simple_func_mlir