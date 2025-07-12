package deep

import (
	"math"
)

// Neuron is a neural network node
type Neuron struct {
	A              Activation
	In             []Synapse
	Out            []Synapse
	Ideal, Desired Deepfloat64
	Sum, Value     Deepfloat64
	Ln             Deepfloat64
	MinSum, MaxSum Deepfloat64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A:      GetActivation(activation),
		MinSum: Deepfloat64(math.MaxFloat64),  // Correct for finding a minimum
		MaxSum: -Deepfloat64(math.MaxFloat64), // Correct for finding a maximum
	}
}

func (n *Neuron) calculateAndFire(refireSynapses bool) {
	n.Sum = 0
	for _, s := range n.In {
		if refireSynapses {
			s.Refire()
		}
		preliminarySum := n.Sum + s.GetOut()
		if !math.IsNaN(float64(preliminarySum)) {
			n.Sum = preliminarySum
		}
	}
	n.Value = n.Activate(n.Sum)

	if n.Sum < n.MinSum {
		n.MinSum = n.Sum
	}
	if n.Sum > n.MaxSum {
		n.MaxSum = n.Sum
	}

	nVal := n.Value
	for _, s := range n.Out {
		s.Fire(nVal)
	}
}

func (n *Neuron) fire() {
	n.calculateAndFire(false)
}

func (n *Neuron) refire() {
	n.calculateAndFire(true)
}

// Activate applies the neurons activation
func (n *Neuron) Activate(x Deepfloat64) Deepfloat64 {
	return n.A.F(x)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x Deepfloat64) Deepfloat64 {
	return n.A.Df(x)
}
