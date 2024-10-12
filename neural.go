package deep

import (
	"fmt"
	//	"sync"
)

// Smallest number
const Eps = 1e-16
const Leps = 1e-20

// Minimal number of iterations
const MinIterations = 5

// Neural is a neural network
type Neural struct {
	Layers     []*Layer
	Config     *Config
	TotalError Deepfloat64
}

// Trainer update mode
type UpdateMode int
type NeuralType int

const (
	// UpdateBottomUp is classic schema updating each
	// layer from the bottom up
	UpdateBottomUp UpdateMode = 0
	// UpdateTopDown is new schema making updates to
	// the top layer until it converge then move to the bottom
	UpdateTopDown UpdateMode = 1

	// Rumelhart type of Network
	RumelhartType NeuralType = 0
	// Kolmogorov type of Network
	KolmogorovType NeuralType = 1
)

// Config defines the network topology, activations, losses etc
type Config struct {
	// Number of inputs
	Inputs int
	// Defines topology:
	// For instance, [5 3 3] signifies a network with two hidden layers
	// containing 5 and 3 nodes respectively, followed an output layer
	// containing 3 nodes.
	Layout []int
	// Defines activation functions per layer
	// Activation functions: {ActivationTanh, ActivationReLU, ActivationSigmoid,
	//	                      ActivationLinear, ActivationSoftmax, ActivationTabulated}
	Activation []ActivationType
	// Defines synapses type per between input and the botton layer and between layers
	Synapse []SynapseType
	// Solver modes: {ModeRegression, ModeBinary, ModeMultiClass, ModeMultiLabel}
	Mode Mode
	// Weight Initialiser type: {NewNormal(σ, μ), NewUniform(σ, μ)}
	Weight WeightType
	// Loss functions: {LossCrossEntropy, LossBinaryCrossEntropy, LossMeanSquared}
	Loss LossType
	// Error/Loss precision
	LossPrecision int
	// Specifies basis size
	Degree int
	// Specify trainer update mode
	TrainerMode UpdateMode
	// Specify network type
	Type NeuralType
	// Specify Synap Tags for the input layer
	InputTags []string
	// Number of training iterations
	Epoch int
	// If Smooth() need to be executed before each training iteration
	Smooth bool
}

// NewNeural returns a new neural network
func NewNeural(c *Config) *Neural {

	if c.Activation == nil {
		c.Activation = make([]ActivationType, len(c.Layout))
		for i := range c.Activation {
			c.Activation[i] = ActivationSigmoid
		}
		c.Activation[len(c.Activation)-1] = OutputActivation(c.Mode)
	}
	if c.Synapse == nil {
		c.Synapse = make([]SynapseType, len(c.Layout))
		for i := range c.Synapse {
			c.Synapse[i] = SynapseTypeAnalytic
		}
	}
	if c.InputTags == nil {
		c.InputTags = make([]string, c.Inputs)
		for i := range c.InputTags {
			c.InputTags[i] = fmt.Sprintf("In:%d", i)
		}
	}
	if c.Loss == LossNone {
		switch c.Mode {
		case ModeMultiClass, ModeMultiLabel:
			c.Loss = LossCrossEntropy
		case ModeBinary:
			c.Loss = LossBinaryCrossEntropy
		default:
			c.Loss = LossMeanSquared
		}
	}
	if c.LossPrecision == 0 {
		c.LossPrecision = 4
	}

	if c.Degree == 0 {
		c.Degree = 2
	}

	layers := initializeLayers(c)

	return &Neural{
		Layers: layers,
		Config: c,
	}
}

func initializeLayers(c *Config) []*Layer {
	var layout []int
	var activation []ActivationType
	var synapse []SynapseType
	if c.Type == KolmogorovType {
		layout = append([]int{2*c.Inputs + 1}, c.Layout...)
		activation = append([]ActivationType{ActivationLinear}, c.Activation...)
		synapse = append([]SynapseType{SynapseTypeAnalytic}, c.Synapse...)
	} else {
		layout = append(layout, c.Layout...)
		activation = append(activation, c.Activation...)
		synapse = append(synapse, c.Synapse...)
	}

	layers := make([]*Layer, len(layout))
	for i := range layers {
		layers[i] = NewLayer(i, layout[i], activation[i], synapse[i])
	}

	layers[0].CreateInputSynapses(c)

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c)
	}

	return layers
}

func (n *Neural) fire() {
	for _, l := range n.Layers {
		l.Fire()
	}
}

// Forward computes a forward pass
func (n *Neural) Forward(input []Deepfloat64) error {
	l := len(input)
	if l != n.Config.Inputs {
		return fmt.Errorf("invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	cl := make(chan struct{})
	ln := len(n.Layers[0].Neurons)

	go func() {
		for j := 0; j < ln/2; j++ {
			neuron := n.Layers[0].Neurons[j]
			cn := make(chan struct{})

			go func() {
				for i := 0; i < l/2; i++ {
					neuron.In[i].Fire(input[i])
				}
				cn <- struct{}{}
			}()

			go func() {
				for i := l / 2; i < l; i++ {
					neuron.In[i].Fire(input[i])
				}
				cn <- struct{}{}
			}()
			<-cn
			<-cn
		}
		cl <- struct{}{}

	}()

	go func() {
		for j := ln / 2; j < ln; j++ {
			neuron := n.Layers[0].Neurons[j]
			cn := make(chan struct{})

			go func() {
				for i := 0; i < l/2; i++ {
					neuron.In[i].Fire(input[i])
				}
				cn <- struct{}{}
			}()

			go func() {
				for i := l / 2; i < l; i++ {
					neuron.In[i].Fire(input[i])
				}
				cn <- struct{}{}
			}()
			<-cn
			<-cn
		}
		cl <- struct{}{}
	}()
	<-cl
	<-cl

	/*
		var wg sync.WaitGroup
		for _, n := range n.Layers[0].Neurons {
			wg.Add(1)
			go func(wg *sync.WaitGroup, n *Neuron) {
				for i := 0; i < len(input); i++ {
					n.In[i].Fire(input[i])
				}
				wg.Done()
			}(&wg, n)
		}
		wg.Wait()
	*/

	n.fire()
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []Deepfloat64) []Deepfloat64 {
	n.Forward(input)

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]Deepfloat64, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Value
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, neuron := range l.Neurons {
			num += len(neuron.In) * (n.Config.Degree + 1)
		}
	}
	return
}

func (n *Neural) String() string {
	var s string
	for _, l := range n.Layers {
		s = fmt.Sprintf("%s\n%v", s, l)
	}
	return s
}

func TotalError(E []Deepfloat64) Deepfloat64 {
	var r Deepfloat64
	for _, x := range E {
		r += x
	}
	return r
}

func (n *Neural) DrawPS(path_prefix string) {
	for _, l := range n.Layers {
		//if l.S != SynapseTypeTabulated {
		//	continue
		//}
		for y, neuron := range l.Neurons {
			for x, in := range neuron.In {
				path := fmt.Sprintf("%sL-%v-N-%v-In-%v.ps", path_prefix, l.Number, y, x)
				in.DrawPS(path)
			}
		}
	}
}

func (n *Neural) Smooth() {
	for _, l := range n.Layers {
		//if l.S != SynapseTypeTabulated {
		//	continue
		//}
		for _, neuron := range l.Neurons {
			for _, in := range neuron.In {
				in.Smooth()
			}
		}
	}
}
