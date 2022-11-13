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
	Weights []Deepfloat64
	In, Out Deepfloat64
	IsBias  bool
	Tag     string
}

// NewSynapse returns a synapse with the weigths set with specified initializer
func NewSynapse(degree int, weight WeightInitializer) *Synapse {
	var weights = make([]Deepfloat64, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &Synapse{Weights: weights}
}

// NewSynapseWithTag returns a synapse with the weigths preset with specified initializer
// and marked with specified tag
func NewSynapseWithTag(tag string, degree int, weight WeightInitializer) *Synapse {
	var weights = make([]Deepfloat64, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &Synapse{Weights: weights, Tag: tag}
}

func (s *Synapse) fire(value Deepfloat64) {
	s.In = value
	mul := Deepfloat64(1)
	s.Out = 0
	for k := 0; k < len(s.Weights); k++ {
		s.Out += s.Weights[k] * mul
		mul *= s.In
	}
}

func (s *Synapse) FireDerivative(value Deepfloat64) Deepfloat64 {
	mul := Deepfloat64(1)
	var res Deepfloat64
	for k := 1; k < len(s.Weights); k++ {
		res += Deepfloat64(k) * mul * s.Weights[k]
		mul *= value
	}
	return res
}

func (s *Synapse) SetTag(tag string) {
	s.Tag = tag
}
