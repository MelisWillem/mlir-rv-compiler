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