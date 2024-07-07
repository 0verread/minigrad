package minigrad

import (
  "fmt"
  "math"
)

type Value struct {
  data float64
  grad float64
  backward func()
  prev []*Value
  op string
}

func NewValue(data float64, children []*Value, op string) *Value {
  return &Value {
    data: data,
    grad: 0,
    backward: func ()  {},
    prev: children,
    op: op,
  }
}

func (v *Value) Add(other interface{}) *Value {
  otherValue, ok := other.(*Value)

  if !ok {
    otherValue = NewValue(other.(float64), nil, "")
  }

  out := NewValue(v.data+otherValue.data, []*Value{v, otherValue}, "+")
  out.backward = func ()  {
    v.grad += out.grad
    otherValue.grad += out.grad
  }

  return out
}

func (v *Value) Mul(other interface{}) *Value {
  otherValue, ok := other.(*Value)

  if !ok {
    otherValue = NewValue(other.(float64), nil, "")
  }

  out := NewValue(v.data*otherValue.data, []*Value{v, otherValue}, "*")
  out.backward = func ()  {
    v.grad += otherValue.data * out.grad
    otherValue.grad +=  v.data * out.grad
  }

  return out
}

func (v *Value) Pow(other float64) *Value {
  out := NewValue(math.Pow(v.data, other), []*Value{v}, fmt.Sprintf("**%v", other))
  out.backward = func ()  {
    v.grad += (other * math.Pow(v.data, other-1)) * out.grad
  }

  return out
}

func (v *Value) Relu() *Value  {
  var outData float64
  if v.data < 0 {
    outData = 0
  } else {
    outData = v.data
  }

  out :=  NewValue(outData, []*Value{v}, "ReLU")
  out.backward = func ()  {
    if out.data > 0 {
      v.grad += out.grad
    }
  }

  return out 
}

func (v *Value) Backward() {
  topo := []*Value{}
  visited := make(map[*Value]bool)

  var buildTopo func(*Value)
  buildTopo = func (v *Value)  {
    if !visited[v] {
      visited[v] = true
      for _, child := range v.prev {
        buildTopo(child)
      }
      topo = append(topo, v)
    }
  }
  buildTopo(v)

  v.grad = 1 
  for i:= len(topo) -1; i >=0; i-- {
    topo[i].backward()
  }
}

func (v *Value) Neg() *Value {
  return v.Mul(-1)
}

func (v *Value) Sub(other interface{}) *Value {
  return v.Add(NewValue(0, nil, "").Sub(other))
}

func (v *Value) Div(other interface{}) *Value {
  return v.Mul(NewValue(1, nil, "").Div(other))
}

func (v *Value) String()  string {
  return fmt.Sprintf("Value(data=%v, grad=%v)", v.data, v.grad)
}

// Helper functions

func Add(other interface{}, v *Value) *Value  {
  return v.Add(other)
}


func Sub(other interface{}, v *Value) *Value  {
  return NewValue(0, nil, "").Sub(v).Add(other)
}

func Mul(other interface{}, v *Value) *Value  {
  return v.Mul(other)
}

func Div(other interface{}, v *Value) *Value  {
  return NewValue(1, nil, "").Div(v).Mul(other)
}


