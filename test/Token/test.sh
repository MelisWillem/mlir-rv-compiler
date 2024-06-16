# RUN: tokenizer %S/simple_func.spl > %t.tokens
# RUN: diff %t.tokens %S/simple_func.tokens