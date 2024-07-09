package minigrad

import (
  "testing"
  "math"
)

func TestNeuron(t *testing.T)  {
  t.Run("NewNeuron", func (t *testing.T)  {
    n := NewNeuron(3, true)
    if len(n.w) != 3 {
      t.Errorf("Expected 3 weights, got %d", len(n.w))
    }
     if n.b == nil {
       t.Error("Expected bias to be non-nil")
     }

     if !n.nonlin {
       t.Error("Expected nonlin to be true")
     }
  })
  t.Run("NeuronCall", func (t *testing.T)  {
    n := NewNeuron(2, true)
    n.w[0] = NewValue(0.5, nil, "")
    n.w[1] = NewValue(-0.5, nil, "")
    n.b = NewValue(0.1, nil, "")

    x := []*Value{NewValue(1, nil, ""), NewValue(2, nil, "")}
    result := n.Call(x)

    expected := 0.1 + 0.5*1 + (-0.5)*2 
    if math.Abs(result.data - math.Max(0, expected)) > 1e-6 {
      t.Errorf("Expected %f, got %f", math.Max(0, expected), result.data)
    }
  })
}

func Testlayer(t *testing.T)  {
  t.Run("NewLayer", func (t *testing.T)  {
    l := NewLayer(3, 2, true)
    if len(l.neurons) != 2 {
      t.Errorf("Expected 2 neurons, got %d", len(l.neurons))
    }
    if len(l.neurons[0].w) != 3 {
      t.Errorf("Expected 3 weights per neuron, got %d", len(l.neurons[0].w))
    }
  })

  t.Run("LayerCall", func (t *testing.T)  {
    l := NewLayer(2, 2, false)
    x := []*Value{NewValue(1, nil, ""), NewValue(2, nil, "")}
    result := l.Call(x)
    if len(result) !=2 {
      t.Errorf("Expected 2 outputs, got %d", len(result))
    }
  })
}

func TestMLP(t *testing.T) {
  t.Run("NewMLP", func (t *testing.T)  {
    m := NewMLP(3, []int{4, 4, 1})
    if len(m.layers) != 3 {
      t.Errorf("Expected 3 layers, got %d", len(m.layers))
    }
  })

  t.Run("MLPCall", func (t *testing.T)  {
    m := NewMLP(2, []int{3, 1})
    x := []*Value{NewValue(1, nil, ""), NewValue(2, nil, "")}
    result := m.Call(x)

    if len(result) != 1 {
      t.Errorf("Expected 1, got %d", len(result))
    }
  })

  t.Run("MLPParameters", func (t *testing.T)  {
    m := NewMLP(2, []int{3, 1})
    params := m.Parameters()
    expectedParamsLen := 2*3 + 3 + 3*1 + 1 
    if len(params) != expectedParamsLen {
      t.Errorf("Expected %d parameters, got %d", expectedParamsLen, len(params))
    }
  })
}


