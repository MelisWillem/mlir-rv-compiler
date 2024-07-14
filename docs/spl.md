# SPL Simple Programming Language

## Why?
SPL is a simple programming language to demonstrate that the backend works. I should never be used by anyone to program anything.

## How to use
```
cspl -i sourceFile.spl
```

## Examples
A function taking in 2 arguments number and flag, or type integer and boolean and returns an int.
```
func foo(number: int, flag: bool) -> int{
    if number > 3 {
        return number;
    }

    if flag {
        return 1;
    }
    return -1;
}
```

for loops:
```
func demo_for_range() {
    var total:int = 1;
    for i in range(0, 5){
        total = total + i;
    }
}
```

```
func demo_list(numbers:int[]) -> int{
    var total:int = 1;
    for n in numbers{
        total = total + n;
    }
    return total;
}
```