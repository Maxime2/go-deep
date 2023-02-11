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
	d_E_y  [][]deep.Deepfloat64
	d_E_x  [][]deep.Deepfloat64
}

func newTraining(layers []*deep.Layer) *internal {
	deltas := make([][]deep.Deepfloat64, len(layers))
	d_E_y := make([][]deep.Deepfloat64, len(layers))
	d_E_x := make([][]deep.Deepfloat64, len(layers))
	for i, l := range layers {
		deltas[i] = make([]deep.Deepfloat64, len(l.Neurons)+1)
		d_E_y[i] = make([]deep.Deepfloat64, len(l.Neurons))
		d_E_x[i] = make([]deep.Deepfloat64, len(l.Neurons))
	}
	return &internal{
		deltas: deltas,
		d_E_y:  d_E_y,
		d_E_x:  d_E_x,
	}
}

// Train trains n
func (t *OnlineTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internal = newTraining(n.Layers)

	//train := make(Examples, len(examples))
	//copy(train, examples)

	t.printer.Init(n)
	numWeights := n.NumWeights()
	t.solver.Init(numWeights)

	ts := time.Now()
	for i := 1; i <= iterations; /*min(iterations, n.Config.N_iterations)*/ i++ {
		//examples.Shuffle()
		t.solver.InitGradients()
		n.Config.N_iterations = deep.MinIterations
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j], i)
		}
		completed := t.adjust(n, i)
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			rCompleted := float64(completed) / float64(numWeights) * 100.0
			t.printer.PrintProgress(n, validation, time.Since(ts), i, rCompleted)
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
		y := neuron.DActivate(neuron.Value)
		//t.deltas[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
		//	neuron.Value,
		//	ideal[i]) * y
		t.d_E_y[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
			neuron.Value,
			ideal[i])
		t.d_E_x[len(n.Layers)-1][i] = t.d_E_y[len(n.Layers)-1][i] * y
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			//var sum deep.Deepfloat64
			var sum_y deep.Deepfloat64
			for k, s := range neuron.Out {
				fd := s.FireDerivative(neuron.Value)
				//sum += fd * t.deltas[i+1][k]
				sum_y += fd * t.d_E_x[i+1][k]
			}
			//sum *= neuron.DActivate(neuron.Value)
			//if !math.IsNaN(float64(sum)) {
			//	t.deltas[i][j] = sum
			//}
			if !math.IsNaN(float64(sum_y)) {
				t.d_E_y[i][j] = sum_y
				t.d_E_x[i][j] = t.d_E_y[i][j] * neuron.DActivate(neuron.Value)
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
					gradient := t.d_E_x[i][j] * deep.Deepfloat64(math.Pow(float64(synapse.Neuron_In.Value), float64(k)))
					t.solver.Update(synapse.Weights[k],
						gradient,
						synapse.In,
						it,
						idx)
					idx++
				}
			}
		}
	}
}

func (t *OnlineTrainer) adjust(n *deep.Neural, it int) int {
	var idx int
	var completed int
	for _, l := range n.Layers {
		for j := range l.Neurons {
			for _, synapse := range l.Neurons[j].In {
				for k := 0; k < len(synapse.Weights); k++ {
					if !synapse.IsComplete[k] {
						update := synapse.Weights[k] + t.solver.Adjust(synapse.Weights[k],
							synapse.Weights_1[k],
							synapse.In,
							it,
							idx)
						if !math.IsNaN(float64(update)) {
							if it > 2 {
								if math.Abs(float64(update-synapse.Weights[k]))/math.Abs(float64(synapse.Weights[k]-synapse.Weights_1[k])) > 1 {
									update = 0
								}
								if (update-synapse.Weights[k])/(1-(update-synapse.Weights[k])/(synapse.Weights[k]-synapse.Weights_1[k])) < deep.Eps {
									//synapse.IsComplete[k] = true
									completed++
								}
							}
							synapse.Weights_1[k] = synapse.Weights[k]
							synapse.Weights[k] = update
						}
					}
					iterations := int(n.Config.Numerator / math.Log(1.0/math.Abs(t.solver.Gradient(idx))))
					if n.Config.N_iterations < iterations {
						n.Config.N_iterations = iterations
					}
					idx++
				}
			}
		}
	}
	return completed
}
