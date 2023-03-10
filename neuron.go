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
	Weights    []Deepfloat64
	Weights_1  []Deepfloat64
	Weights_2  []Deepfloat64
	IsComplete []bool
	FakeRoot   [][]Deepfloat64
	In, Out    Deepfloat64
	IsBias     bool
	Tag        string
}

// NewSynapse returns a synapse with the weigths set with specified initializer
func NewSynapse(degree int, weight WeightInitializer) *Synapse {
	var weights = make([]Deepfloat64, degree+1)
	var weights_1 = make([]Deepfloat64, degree+1)
	var weights_2 = make([]Deepfloat64, degree+1)
	var isComplete = make([]bool, degree+1)
	var fakeRoot = make([][]Deepfloat64, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &Synapse{
		Weights:    weights,
		Weights_1:  weights_1,
		Weights_2:  weights_2,
		IsComplete: isComplete,
		FakeRoot:   fakeRoot,
		In:         0,
		Out:        0,
		IsBias:     false,
		Tag:        "",
	}
}

// NewSynapseWithTag returns a synapse with the weigths preset with specified initializer
// and marked with specified tag
func NewSynapseWithTag(tag string, degree int, weight WeightInitializer) *Synapse {
	var weights = make([]Deepfloat64, degree+1)
	var weights_1 = make([]Deepfloat64, degree+1)
	var weights_2 = make([]Deepfloat64, degree+1)
	var isComplete = make([]bool, degree+1)
	var fakeRoot = make([][]Deepfloat64, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &Synapse{
		Weights:    weights,
		Weights_1:  weights_1,
		Weights_2:  weights_2,
		IsComplete: isComplete,
		FakeRoot:   fakeRoot,
		In:         0,
		Out:        0,
		IsBias:     false,
		Tag:        tag,
	}
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
	for _, root := range s.FakeRoot[k] {
		f /= (value - root)
	}
	return f
}

func (s *Synapse) AddFakeRoot(k int, root Deepfloat64) {
	s.FakeRoot[k] = append(s.FakeRoot[k], root)
}

func (s *Synapse) ClearFakeRoots(k int) {
	s.FakeRoot[k] = s.FakeRoot[k][:0]
}

func (s *Synapse) GetGradient(D_E_x Deepfloat64, k int) Deepfloat64 {
	return D_E_x * s.WeightFunction(Deepfloat64(math.Pow(float64(s.In), float64(k))), k)
}
