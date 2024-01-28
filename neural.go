package deep

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/theothertomelliott/acyclic"
)

// Smallest number
const Eps = 1e-16
const Leps = 1e-20

const Modulus = 1

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
}

// NewNeural returns a new neural network
func NewNeural(c *Config) *Neural {

	if c.Activation == nil {
		c.Activation = make([]ActivationType, len(c.Layout))
		for i := range c.Activation {
			c.Activation[i] = ActivationSigmoid
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
	var act []ActivationType
	var activation ActivationType
	if c.Type == KolmogorovType {
		layout = append([]int{2*c.Inputs + 1}, c.Layout...)
		act = append([]ActivationType{ActivationSigmoid}, c.Activation...)
	} else {
		layout = append(layout, c.Layout...)
		act = append(act, c.Activation...)
	}

	layers := make([]*Layer, len(layout))
	for i := range layers {
		if i == (len(layers)-1) && c.Mode != ModeDefault {
			activation = OutputActivation(c.Mode)
		} else {
			activation = act[i%len(act)]
		}
		layers[i] = NewLayer(i, layout[i], activation)
	}

	A := Modulus / (float64(c.Inputs)) / float64(len(layers[0].Neurons))
	wA := Deepfloat64(0)
	wi := GetWeightFunction(c.Weight, Eps, A)
	for _, neuron := range layers[0].Neurons {
		neuron.In = make([]*Synapse, c.Inputs)

		if c.InputTags == nil {
			for i := range neuron.In {
				neuron.In[i] = NewSynapseWithTag(neuron, c.Degree, wi, fmt.Sprintf("In:%d", i))
				neuron.In[i].SetWeight(0, wA)
				wA += neuron.In[i].GetWeight(1)
			}
		} else {
			for i := range neuron.In {
				neuron.In[i] = NewSynapseWithTag(neuron, c.Degree, wi, c.InputTags[i])
				neuron.In[i].SetWeight(0, wA)
				wA += neuron.In[i].GetWeight(1)
			}
		}
	}

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c.Degree, c.Weight)
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
	if len(input) != n.Config.Inputs {
		return fmt.Errorf("Invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, n := range n.Layers[0].Neurons {
		for i := 0; i < len(input); i++ {
			n.In[i].Fire(input[i])
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

// Save the network in NET format
func (n *Neural) Net(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for l, lr := range n.Layers {
		fmt.Fprintf(f, "L: %d\n", l)
		for n, nr := range lr.Neurons {
			fmt.Fprintf(f, "  N: %d;  Sum: %v; Value: %v;\n", n, nr.Sum, nr.Value)
			for _, in := range nr.In {
				fmt.Fprintf(f, " [%v %v]", in.In, in.Out)
			}
			fmt.Fprintf(f, "\n")
		}
	}

	fmt.Fprintf(f, "\n")

	return nil
}

func TotalError(E []Deepfloat64) Deepfloat64 {
	var r Deepfloat64
	for _, x := range E {
		r += x
	}
	return r
}
