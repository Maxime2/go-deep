package deep

import (
	"math"
)

// Neuron is a neural network node
type Neuron struct {
	A     ActivationType
	In    []*Synapse
	Out   []*Synapse
	Value Deepfloat64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) fire() {
	var sum Deepfloat64
	for _, s := range n.In {
		preliminarySum := sum + s.Out
		if !math.IsNaN(float64(preliminarySum)) {
			sum = preliminarySum
		}
	}
	n.Value = n.Activate(sum)

	nVal := n.Value
	for _, s := range n.Out {
		s.fire(nVal)
	}
}

// Activate applies the neurons activation
func (n *Neuron) Activate(x Deepfloat64) Deepfloat64 {
	return GetActivation(n.A).F(x)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x Deepfloat64) Deepfloat64 {
	return GetActivation(n.A).Df(x)
}

// Synapse is an edge between neurons
type Synapse struct {
	Weight0, Weight1, Weight2 Deepfloat64
	In, Out                   Deepfloat64
	IsBias                    bool
}

// NewSynapse returns a synapse with the specified initialized weight
func NewSynapse(weight0, weight1, weight2 Deepfloat64) *Synapse {
	return &Synapse{Weight0: weight0, Weight1: weight1, Weight2: weight2}
}

func (s *Synapse) fire(value Deepfloat64) {
	s.In = value
	s.Out = s.Weight0 + s.In*s.Weight1 + s.In*s.Weight2*s.In
}
