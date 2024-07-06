# minigrad
Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd) implementation in Go


## Usage
```go
package main

import (
  mg "github.com/0verread/minigrad"
)

func main() {
  // inputs a, b
  a := mg.NewValue(2.0)
  b := mg.NewValue(0.1)

  x := mg.NewValue(-2.0)
  y := mg.NewValue(1.0)

  bi := mg.New(4.908765)

  xa := mg.Mul(x, a)
  yb := mg.Mul(y, b)
}
```
