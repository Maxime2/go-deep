package deep

import "fmt"

// Layer is a set of neurons and corresponding activation
type Layer struct {
	Number  int
	Neurons []*Neuron
	A       ActivationType
}

// NewLayer creates a new layer with n nodes
func NewLayer(l, n int, activation ActivationType) *Layer {
	neurons := make([]*Neuron, n)

	for i := 0; i < n; i++ {
		neurons[i] = NewNeuron(activation)
	}
	return &Layer{
		Number:  l,
		Neurons: neurons,
		A:       activation,
	}
}

func (l *Layer) Fire() {
	for _, n := range l.Neurons {
		n.fire()
	}
	if l.A == ActivationSoftmax {
		outs := make([]Deepfloat64, len(l.Neurons))
		for i, neuron := range l.Neurons {
			outs[i] = neuron.Value
		}
		sm := Softmax(outs)
		for i, neuron := range l.Neurons {
			neuron.Value = Deepfloat64(sm[i])
		}
	}
}

func (l *Layer) Refire() {
	for _, n := range l.Neurons {
		n.refire()
	}
	if l.A == ActivationSoftmax {
		outs := make([]Deepfloat64, len(l.Neurons))
		for i, neuron := range l.Neurons {
			outs[i] = neuron.Value
		}
		sm := Softmax(outs)
		for i, neuron := range l.Neurons {
			neuron.Value = Deepfloat64(sm[i])
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
func (l *Layer) Connect(next *Layer, degree int, weight WeightType) {
	num_neurons := len(l.Neurons)
	A := 1.0 / (float64(degree+1) * float64(num_neurons) * float64(len(next.Neurons)+1))
	num_neurons = (num_neurons + 1) / 2
	for i := range l.Neurons {
		wi := GetWeightFunction(weight, A/2.0, 24.0*float64(i+1-num_neurons)*A)
		for j, neuron := range next.Neurons {
			syn := NewSynapseWithTag(neuron, degree, wi, fmt.Sprintf("L:%d N:%d", l.Number, i))
			x := len(next.Neurons[j].In)
			if x > 0 {
				syn.SetWeight(0, next.Neurons[j].In[x-1].GetWeight(0)+next.Neurons[j].In[x-1].GetWeight(1))
			} else {
				syn.SetWeight(0, 0)
			}
			l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
			next.Neurons[j].In = append(next.Neurons[j].In, syn)
		}
	}
}

// ApplyBias creates and returns a bias synapse for each neuron in l
func (l *Layer) ApplyBias(degree int, weight WeightInitializer) []*Synapse {
	biases := make([]*Synapse, len(l.Neurons))
	for i, neuron := range l.Neurons {
		biases[i] = NewSynapseWithTag(neuron, degree, weight, fmt.Sprintf("L:%d B:%d", l.Number, i))
		biases[i].IsBias = true
		l.Neurons[i].In = append(l.Neurons[i].In, biases[i])
	}
	return biases
}

func (l Layer) String() string {
	weights := make([][][]Deepfloat64, len(l.Neurons))
	for i, n := range l.Neurons {
		weights[i] = make([][]Deepfloat64, len(n.In))
		for j, s := range n.In {
			weights[i][j] = s.Weights
		}
	}
	return fmt.Sprintf("%+v", weights)
}

func (l Layer) NumIns() (num int) {
	for _, neuron := range l.Neurons {
		num += len(neuron.In)
	}
	return
}
