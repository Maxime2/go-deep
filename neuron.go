package deep

import (
	"math"
)

// Neuron is a neural network node
type Neuron struct {
	A              ActivationType
	In             []*Synapse
	Out            []*Synapse
	Ideal, Desired Deepfloat64
	Sum, Value     Deepfloat64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) fire() {
	n.Sum = 0
	for _, s := range n.In {
		preliminarySum := n.Sum + s.Out
		if !math.IsNaN(float64(preliminarySum)) {
			n.Sum = preliminarySum
		}
	}
	n.Value = n.Activate(n.Sum)

	nVal := n.Value
	for _, s := range n.Out {
		s.Fire(nVal)
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
	Weights    []Deepfloat64
	Weights_1  []Deepfloat64
	IsComplete []bool
	Up         *Neuron
	In, Out    Deepfloat64
	IsBias     bool
	Tag        string
}

// NewSynapse returns a synapse with the weigths set with specified initializer
func NewSynapse(up *Neuron, degree int, weight WeightInitializer) *Synapse {
	var weights = make([]Deepfloat64, degree+1)
	var weights_1 = make([]Deepfloat64, degree+1)
	var isComplete = make([]bool, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &Synapse{
		Weights:    weights,
		Weights_1:  weights_1,
		IsComplete: isComplete,
		In:         0,
		Out:        0,
		IsBias:     false,
		Tag:        "",
		Up:         up,
	}
}

// NewSynapseWithTag returns a synapse with the weigths preset with specified initializer
// and marked with specified tag
func NewSynapseWithTag(up *Neuron, degree int, weight WeightInitializer, tag string) *Synapse {
	var weights = make([]Deepfloat64, degree+1)
	var weights_1 = make([]Deepfloat64, degree+1)
	var isComplete = make([]bool, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &Synapse{
		Weights:    weights,
		Weights_1:  weights_1,
		IsComplete: isComplete,
		In:         0,
		Out:        0,
		IsBias:     false,
		Tag:        tag,
		Up:         up,
	}
}

func (s *Synapse) Fire(value Deepfloat64) {
	s.In = value
	mul := Deepfloat64(1)
	s.Out = 0
	for k := 0; k < len(s.Weights); k++ {
		s.Out += s.Weights[k] * mul
		mul *= s.In
	}
}

func (s *Synapse) FireDerivative() Deepfloat64 {
	mul := Deepfloat64(1)
	var res Deepfloat64
	for k := 1; k < len(s.Weights); k++ {
		res += Deepfloat64(k) * mul * s.Weights[k]
		mul *= s.In
	}
	return res
}

func (s *Synapse) SetTag(tag string) {
	s.Tag = tag
}

func (s *Synapse) WeightFunction(value Deepfloat64, k int) Deepfloat64 {
	f := value
	//for _, root := range s.FakeRoot[k] {
	//	f /= (value - root)
	//}
	return f
}

func (s *Synapse) GetGradient(D_E_x Deepfloat64, k int) Deepfloat64 {
	return D_E_x * s.WeightFunction(Deepfloat64(math.Pow(float64(s.In), float64(k))), k)
}
