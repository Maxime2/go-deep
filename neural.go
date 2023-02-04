package deep

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"

	"github.com/theothertomelliott/acyclic"
)

// Smallest number
const Eps = 1e-12

// Minimal number of iterations
const MinIterations = 5

// Neural is a neural network
type Neural struct {
	Layers []*Layer
	Biases [][]*Synapse
	Config *Config
}

// Config defines the network topology, activations, losses etc
type Config struct {
	// Number of inputs
	Inputs int
	// Defines topology:
	// For instance, [5 3 3] signifies a network with two hidden layers
	// containing 5 and 3 nodes respectively, followed an output layer
	// containing 3 nodes.
	Layout []int
	// Activation functions: {ActivationTanh, ActivationReLU, ActivationSigmoid}
	Activation ActivationType
	// Solver modes: {ModeRegression, ModeBinary, ModeMultiClass, ModeMultiLabel}
	Mode Mode
	// Initializer for weights: {NewNormal(σ, μ), NewUniform(σ, μ)}
	Weight WeightInitializer `json:"-"`
	// Loss functions: {LossCrossEntropy, LossBinaryCrossEntropy, LossMeanSquared}
	Loss LossType
	// Apply bias nodes
	Bias bool
	// Error/Loss precision
	LossPrecision int
	// Specifies basis size
	Degree int
	// Specify Synap Tags for the input layer
	InputTags []string
	// Estimate for number of iterations needed
	// for internal use
	N_iterations int
	// Precision for weights calculation
	// for internal use
	Numerator float64
}

// NewNeural returns a new neural network
func NewNeural(c *Config) *Neural {

	if c.Weight == nil {
		c.Weight = NewUniform(0.5, 0)
	}
	if c.Activation == ActivationNone {
		c.Activation = ActivationSigmoid
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

	if c.Numerator == 0 {
		c.Numerator = math.Log(math.Pow10(c.LossPrecision))
	}

	c.N_iterations = MinIterations // int(c.Numerator)

	layers := initializeLayers(c)

	var biases [][]*Synapse
	if c.Bias {
		biases = make([][]*Synapse, len(layers))
		for i := 0; i < len(layers); i++ {
			if c.Mode == ModeRegression && i == len(layers)-1 {
				continue
			}
			biases[i] = layers[i].ApplyBias(c.Degree, c.Weight)
		}
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: c,
	}
}

func initializeLayers(c *Config) []*Layer {
	layers := make([]*Layer, len(c.Layout))
	for i := range layers {
		act := c.Activation
		if i == (len(layers)-1) && c.Mode != ModeDefault {
			act = OutputActivation(c.Mode)
		}
		layers[i] = NewLayer(i, c.Layout[i], act)
	}

	for _, neuron := range layers[0].Neurons {
		neuron.In = make([]*Synapse, c.Inputs)
		if c.InputTags == nil {
			for i := range neuron.In {
				neuron.In[i] = NewSynapseWithTag(fmt.Sprintf("In:%d", i), neuron, c.Degree, c.Weight)
			}
		} else {
			for i := range neuron.In {
				neuron.In[i] = NewSynapseWithTag(c.InputTags[i], neuron, c.Degree, c.Weight)
			}
		}
	}

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c.Degree, c.Weight)
	}

	return layers
}

func (n *Neural) fire() {
	for _, b := range n.Biases {
		for _, s := range b {
			s.fire(1)
		}
	}
	for _, l := range n.Layers {
		l.fire()
	}
}

// Forward computes a forward pass
func (n *Neural) Forward(input []Deepfloat64) error {
	if len(input) != n.Config.Inputs {
		return fmt.Errorf("Invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, n := range n.Layers[0].Neurons {
		for i := 0; i < len(input); i++ {
			n.In[i].fire(input[i])
		}
	}
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
		s = fmt.Sprintf("%s\n%s", s, l)
	}
	return s
}

// Save saves network in readable JSON into the file specified
func (n *Neural) SaveReadable(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	acyclic.Fprint(f, n)
	return nil
}

// Save saves network into the file specified to be loaded later
func (n *Neural) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := n.Marshal()
	if err != nil {
		return err
	}
	_, err = io.Copy(f, bytes.NewReader(b))
	return err
}

// Load retrieves network from the file specified created using Save method
func Load(path string) (*Neural, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, _ := ioutil.ReadAll(f)
	return Unmarshal(b)
}

// Save the network in DOT format for graphviz
func (n *Neural) Dot(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "digraph {\n")

	for l, lr := range n.Layers {
		for n, nr := range lr.Neurons {
			for _, in := range nr.In {
				fmt.Fprintf(f, "\"%s\" -> \"L:%d N:%d\"[label=\"%v\"]\n", in.Tag, l, n, in.Weights)
			}
		}
	}

	fmt.Fprintf(f, "}\n")

	return nil
}
