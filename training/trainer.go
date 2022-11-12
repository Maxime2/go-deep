package training

import (
	"math"
	"time"

	deep "github.com/Maxime2/go-deep"
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
	deltas [][]deep.Deepfloat64
}

func newTraining(layers []*deep.Layer) *internal {
	deltas := make([][]deep.Deepfloat64, len(layers))
	for i, l := range layers {
		deltas[i] = make([]deep.Deepfloat64, len(l.Neurons)+1)
	}
	return &internal{
		deltas: deltas,
	}
}

// Train trains n
func (t *OnlineTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internal = newTraining(n.Layers)

	//train := make(Examples, len(examples))
	//copy(train, examples)

	t.printer.Init(n)
	t.solver.Init(n.NumWeights())

	ts := time.Now()
	for i := 1; i <= iterations; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j], i)
		}
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(n, validation, time.Since(ts), i)
		}
	}
}

func (t *OnlineTrainer) learn(n *deep.Neural, e Example, it int) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n, it)
}

func (t *OnlineTrainer) calculateDeltas(n *deep.Neural, ideal []deep.Deepfloat64) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.deltas[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
			neuron.Value,
			ideal[i],
			neuron.DActivate(neuron.Value))
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum deep.Deepfloat64
			for k, s := range neuron.Out {
				sum += s.FireDerivative(neuron.Value) * t.deltas[i+1][k]
			}
			sum *= neuron.DActivate(neuron.Value)
			if !math.IsNaN(float64(sum)) {
				t.deltas[i][j] = sum
			}
		}
	}
}

func (t *OnlineTrainer) update(n *deep.Neural, it int) {
	var idx int
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for _, synapse := range l.Neurons[j].In {
				for k := 0; k < len(synapse.Weights); k++ {
					update := synapse.Weights[k] + t.solver.Update(synapse.Weights[k],
						t.deltas[i][j]*deep.Deepfloat64(math.Pow(float64(synapse.In), float64(k))),
						synapse.In,
						it,
						idx)
					if !math.IsNaN(float64(update)) {
						synapse.Weights[k] = update
					}
				}
				idx++
			}
		}
	}
}
