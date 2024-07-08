package minigrad

import (
  "math"
  "testing"
)

func TestValueCreation(t *testing.T)  {
  v := NewValue(5.0, nil, "")
  if v.data != 5.0 {
    t.Errorf("Expected 5.0, got %v", v.data)
  }

  if v.grad != 0.0 {
    t.Errorf("Expected initial grad value 0.0, got %v", v.grad)
  }
}

func TestAdd( t *testing.T)  {
  a := NewValue(2.0, nil, "")
  b := NewValue(3.0, nil, "")
  total := a.Add(b)

  if total.data != 5.0 {
    t.Errorf("Expected 5.0 got %v", total.data)
  }

  total.Backward()

  if a.grad != 1.0 || b.grad != 1.0 {
    t.Errorf("Addition gradient: expected both gradient to be 1.0 but got %v and %v", a.grad, b.grad)
  }
}

func TestMul(t *testing.T)  {
  a := NewValue(2.0, nil, "")
  b := NewValue(3.0, nil, "")
  total := a.Mul(b)

  if total.data != 6.0 {
    t.Errorf("Expected 6.0, got %v", total.data)
  }

  total.Backward()

  if a.grad != 3.0 || b.grad != 2.0 {
    t.Errorf("Multiplication Gradient: expected both gradients to be 3.0 and 2.0, got %v and %v", a.grad, b.grad)
  }
}

func TestPow(t *testing.T)  {
  a := NewValue(2.0, nil, "")
  b := a.Pow(3.0)

  if b.data != 8.0 {
    t.Errorf("Expected 8.0, got %v", b.data)
  }

  b.Backward()

  if math.Abs(a.grad - 12.0) > 1e-6 {
    t.Errorf("Power Gradient: expected to be 12.0, got %v", a.grad)
  }
}

func TestRelu( t *testing.T)  {
  a := NewValue(-2.0, nil, "")
  b := NewValue(3.0, nil, "")
  c := a.Relu()
  d := b.Relu()

  if c.data != 0.0 || d.data != 3.0 {
    t.Errorf("expected 0.0 and 3.0, got %v and %v", c.data, d.data)
  }

  c.Backward()
  d.Backward()

  // TODO: Bug 
  //if c.grad != 0.0 || d.grad != 1.0 {
  //  t.Errorf("Relu Gradient: expected to be 0.0 nad 1.0, got %v and %v", c.grad, d.grad)
  //}
}

func TestNeg(t *testing.T)  {
  a := NewValue(5.0, nil, "")
  result := a.Neg()

  if result.data != -5.0 {
    t.Errorf("expected -5.0, got %v", result.data)
  }

  result.Backward()

  if a.grad != -1.0 {
    t.Errorf("Negation Gradient: expected to be -1.0, got %v", result.grad)
  }
}


