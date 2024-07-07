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
	a := mg.NewValue(-4.0, nil, "")
	b := mg.NewValue(2.0, nil, "")
	c := a.Add(b)
	d := a.Mul(b).Add(b.Pow(3))
	c = c.Add(c.Add(mg.NewValue(1.0, nil, "")))
	c = c.Add(mg.NewValue(1.0, nil, "").Add(c))
	d = d.Add(d.Mul(2.0).Add(b.Add(a).Relu()))
  
  d.Backward()

}
```

