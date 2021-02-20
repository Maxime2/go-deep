package deep

import (
	"math"
)

const synapsePoolSize int = 10
const eThr float64 = float64(1) / float64(synapsePoolSize)

// Neuron is a neural network node
type Neuron struct {
	A     ActivationType
	In    []*Synapse
	Out   []*Synapse
	Value float64
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation ActivationType) *Neuron {
	return &Neuron{
		A: activation,
	}
}

func (n *Neuron) fire() {
	var sum float64
	for _, s := range n.In {
		if !math.IsNaN(sum + s.Out) {
			sum += s.Out
		}
	}
	n.Value = n.Activate(sum)

	nVal := n.Value
	for _, s := range n.Out {
		s.fire(nVal)
	}
}

// Activate applies the neurons activation
func (n *Neuron) Activate(x float64) float64 {
	return GetActivation(n.A).F(x)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x float64) float64 {
	return GetActivation(n.A).Df(x)
}

// Synapse is an edge between neurons
type Synapse struct {
	Weight, In               [synapsePoolSize]float64
	Out                      float64
	currentIn, currentWeight float64
	mi                       int
	IsBias                   bool
}

// NewSynapse returns a synapse with the specified initialized weight
func NewSynapse(c *Config) *Synapse {
	w := [synapsePoolSize]float64{}
	for i := range w {
		w[i] = c.Weight()
	}
	return &Synapse{Weight: w}
}

func (s *Synapse) fire(value float64) {
	var mj, mi int
	mj = 1
	diff := math.Abs(value - s.In[mi])
	for i := range s.In {
		if math.Abs(value-s.In[i]) <= diff {
			mi = i
		}
	}
	if diff < eThr {
		s.In[mi] = (s.In[mi] + value) / 2
	} else {
		mi = 0
		mj = 1
		W := s.Weight[mi]
		diff = math.Abs(s.In[mi] - s.In[mj])
		for i := 0; i < len(s.In)-1; i++ {
			for j := i + 1; j < len(s.In); j++ {
				vdiff := math.Abs(s.In[i] - s.In[j])
				if vdiff < diff {
					diff = vdiff
					mi = i
					mj = j
				}
			}
		}
		s.In[mj] = (s.In[mi] + s.In[mj]) / 2
		s.Weight[mj] = (s.Weight[mi] + s.Weight[mj]) / 2
		s.In[mi] = value
		s.Weight[mi] = W
	}

	//s.In = value
	s.Out = s.In[mi] * s.Weight[mi]
	s.currentIn = s.In[mi]
	s.currentWeight = s.Weight[mi]
	s.mi = mi
}

// GetIn returns current value of Synapse's input
func (s *Synapse) GetIn() float64 {
	return s.currentIn
}

// GetWeight returns current value of Synapse's weight
func (s *Synapse) GetWeight() float64 {
	return s.currentWeight
}

// UpdateWeight updates current Synapse's weight
func (s *Synapse) UpdateWeight(weight float64) {
	s.Weight[s.mi] = weight
	s.currentWeight = weight
}
