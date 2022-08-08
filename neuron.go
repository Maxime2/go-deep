package deep

import (
	"math"
)

// Neuron is a neural network node
type Neuron struct {
	A     ActivationType
	In    []*Synapse
	Out   []*Synapse
	Value deepfloat64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) fire() {
	var sum deepfloat64
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
func (n *Neuron) Activate(x deepfloat64) deepfloat64 {
	return GetActivation(n.A).F(x)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x deepfloat64) deepfloat64 {
	return GetActivation(n.A).Df(x)
}

// Synapse is an edge between neurons
type Synapse struct {
	Weight0, Weight1, Weight2 deepfloat64
	In, Out                   deepfloat64
	IsBias                    bool
}

// NewSynapse returns a synapse with the specified initialized weight
func NewSynapse(weight deepfloat64) *Synapse {
	return &Synapse{Weight0: weight, Weight1: weight, Weight2: weight}
}

func (s *Synapse) fire(value deepfloat64) {
	s.In = value
	s.Out = s.Weight0 + s.In*s.Weight1 + s.In*s.Weight2*s.In
}
