package training

import (
//	"fmt"
	"math"
	"time"

	deep "github.com/patrikeh/go-deep"
)

// Trainer is a neural network trainer
type Trainer interface {
	Train(n *deep.Neural, examples, validation Examples, iterations int)
}

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	solver    Solver
	printer   *StatsPrinter
	verbosity int
}

// NewTrainer creates a new trainer
func NewTrainer(solver Solver, verbosity int) *OnlineTrainer {
	return &OnlineTrainer{
		solver:    solver,
		printer:   NewStatsPrinter(),
		verbosity: verbosity,
	}
}

type internal struct {
	deltas [][]float64
}

func newTraining(layers []*deep.Layer) *internal {
	deltas := make([][]float64, len(layers))
	for i, l := range layers {
		deltas[i] = make([]float64, len(l.Neurons) + 1)
	}
	return &internal{
		deltas: deltas,
	}
}

// Train trains n
func (t *OnlineTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internal = newTraining(n.Layers)

	train := make(Examples, len(examples))
	copy(train, examples)

	t.printer.Init(n)
	t.solver.Init(n.NumWeights())

	ts := time.Now()
	for i := 1; i <= iterations; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			//			fmt.Println("j=", j, "example=", examples[j])
			t.learn(n, examples[j], i)
		}
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(n, validation, time.Since(ts), i)
		}
	}
}

// AdjustInWeights adjusts weights of input layer to normalise input signal
func AdjustInWeights(n *deep.Neural, examples Examples) {
	A := make([]float64, len(examples[0].Input))
	for i := 0; i < len(examples[0].Input); i++ {
		A[i] = math.Abs(examples[0].Input[i])
	}
	for j := 1; j < len(examples); j++ {
		for i := 0; i < len(examples[j].Input); i++ {
			B := math.Abs(examples[j].Input[i])
			if B > A[i] {
				A[i] = B
			}
		}
	}
	for _, n := range n.Layers[0].Neurons {
		for i := 0; i < len(A); i++ {
			if A[i] > 0.99 {
				n.In[i].Weight = n.In[i].Weight / A[i]
			}
		}
	}
}

func (t *OnlineTrainer) learn(n *deep.Neural, e Example, it int) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n, it)
}

func (t *OnlineTrainer) calculateDeltas(n *deep.Neural, ideal []float64) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.deltas[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
			neuron.Value,
			ideal[i],
			neuron.DActivate(neuron.Value))
		//fmt.Println("neuron.Value=", neuron.Value, ".DActivate=", neuron.DActivate(neuron.Value))
		//fmt.Println("i", i, "ideal", ideal[i])
		//fmt.Println("deltas=", t.deltas[len(n.Layers)-1][i])
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * t.deltas[i+1][k]
			}
			if math.IsNaN(sum) {
				t.deltas[i][j] = neuron.DActivate(neuron.Value)
			} else {
				t.deltas[i][j] = neuron.DActivate(neuron.Value) * sum
			}
		}
	}
}

func (t *OnlineTrainer) update(n *deep.Neural, it int) {
	var idx int
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				update := t.solver.Update(l.Neurons[j].In[k].Weight,
					t.deltas[i][j]*l.Neurons[j].In[k].In,
					l.Neurons[j].In[k].In,
					it,
					idx)
				if !math.IsNaN(l.Neurons[j].In[k].Weight + update) {
					l.Neurons[j].In[k].Weight += update
				}
				idx++
			}
		}
	}
}
