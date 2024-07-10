# minigrad
Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd) implementation in Go


## Usage
```go
package main

import (
  mg "github.com/0verread/minigrad"
)

func main() {
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

## Contribution

Feel free to raise a issue or PR to make enhancements or report a bug. I would 
love to see more test cases.

## License

minigrad is under [MIT](https://github.com/0verread/LICENSE).
