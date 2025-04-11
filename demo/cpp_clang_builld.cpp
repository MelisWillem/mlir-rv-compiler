extern "C" {

__attribute__((noinline))
int bar(int a, int b){
  return a + b;
}

int foo(int a, int b){
  return bar(a+1, b);
}

}
