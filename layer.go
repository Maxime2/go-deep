package deep

import "fmt"

// Layer is a set of neurons and corresponding activation
type Layer struct {
	Number  int
	A       ActivationType
	S       SynapseType
	Neurons []*Neuron
}

// NewLayer creates a new layer with n nodes
func NewLayer(l, n int, activation ActivationType, synapse SynapseType) *Layer {
	//func NewLayer(c *Config, l int) *Layer {
	//	n := c.Layout[l]
	//	activation := c.Activation[l]
	//	synapse := c.Synapse[l]

	neurons := make([]*Neuron, n)

	for i := 0; i < n; i++ {
		neurons[i] = NewNeuron(activation)
	}
	return &Layer{
		Number:  l,
		Neurons: neurons,
		A:       activation,
		S:       synapse,
	}
}

func (l *Layer) Fire() {
	ln := len(l.Neurons)
	cl := make(chan struct{})

	go func() {
		for j := 0; j < ln/2; j++ {
			l.Neurons[j].fire()
		}
		cl <- struct{}{}
	}()

	go func() {
		for j := ln / 2; j < ln; j++ {
			l.Neurons[j].fire()
		}
		cl <- struct{}{}
	}()
	<-cl
	<-cl

	/*
		for _, n := range l.Neurons {
			n.fire()
		}
	*/

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

// CreateInputSynapses create input synapses for the bottom layer
func (l *Layer) CreateInputSynapses(c *Config) {
	switch l.S {
	case SynapseTypeTabulated:
		for _, neuron := range l.Neurons {
			neuron.In = make([]Synapse, c.Inputs)
			for i := range neuron.In {
				neuron.In[i] = NewSynapseTabulated(neuron, c.InputTags[i])
			}
		}
	case SynapseTypeAnalytic:
		domain_min, domain_max := GetActivation(l.A).Domain()
		A := float64(2*(domain_max-domain_min)) / float64(c.Inputs) / float64(c.Inputs) / float64(len(l.Neurons)) / float64(c.Degree+1)
		wA := Deepfloat64(domain_min)
		wi := GetWeightFunction(c.Weight, A/20, A)
		wEps := Deepfloat64(A / 50 / float64(c.Inputs))
		for _, neuron := range l.Neurons {
			neuron.In = make([]Synapse, c.Inputs)
			for i := range neuron.In {
				neuron.In[i] = NewSynapseAnalytic(neuron, c.Degree, wi, c.InputTags[i])
				neuron.In[i].SetWeight(0, wA)
				wA += neuron.In[i].GetWeight(1) + wEps
			}
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
// func (l *Layer) Connect(next *Layer, degree int, weight WeightType) {
func (l *Layer) Connect(next *Layer, c *Config) {
	switch next.S {
	case SynapseTypeTabulated:
		for j, neuron := range next.Neurons {
			for i := range l.Neurons {
				syn := NewSynapseTabulated(neuron, fmt.Sprintf("L:%d N:%d", l.Number, i))
				l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
				next.Neurons[j].In = append(next.Neurons[j].In, syn)
			}
		}
	case SynapseTypeAnalytic:
		weight := c.Weight
		if c.Type == KolmogorovType {
			weight = WeightIdentity
		}
		num_neurons := len(l.Neurons)
		domain_min, domain_max := next.Neurons[0].A.Domain()
		A := float64(2*(domain_max-domain_min)) / float64(num_neurons) / float64(num_neurons) / float64(len(next.Neurons)) / float64(c.Degree+1)
		wA := Deepfloat64(domain_min)
		wi := GetWeightFunction(weight, A/20, A)
		wEps := Deepfloat64(A / 50 / float64(num_neurons))
		for j, neuron := range next.Neurons {
			for i := range l.Neurons {
				syn := NewSynapseAnalytic(neuron, c.Degree, wi, fmt.Sprintf("L:%d N:%d", l.Number, i))
				if weight != WeightIdentity {
					syn.SetWeight(0, wA)
					wA += syn.GetWeight(1) + wEps
				}
				l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
				next.Neurons[j].In = append(next.Neurons[j].In, syn)
			}
		}
	}
}

// ApplyBias creates and returns a bias synapse for each neuron in l
func (l *Layer) ApplyBias(degree int, weight WeightInitializer) []Synapse {
	biases := make([]Synapse, len(l.Neurons))
	for i, neuron := range l.Neurons {
		biases[i] = NewSynapseAnalytic(neuron, degree, weight, fmt.Sprintf("L:%d B:%d", l.Number, i))
		biases[i].SetIsBias(true)
		l.Neurons[i].In = append(l.Neurons[i].In, biases[i])
	}
	return biases
}

//func (l Layer) String() string {
//	weights := make([][][]Deepfloat64, len(l.Neurons))
//	for i, n := range l.Neurons {
//		weights[i] = make([][]Deepfloat64, len(n.In))
//		for j, s := range n.In {
//			weights[i][j] = s.Weights
//		}
//	}
//	return fmt.Sprintf("%+v", l)
//}

func (l Layer) NumIns() (num int) {
	for _, neuron := range l.Neurons {
		num += len(neuron.In)
	}
	return
}
