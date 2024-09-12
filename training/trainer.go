package training

import (
	"fmt"
	"math"
	"os"
	"time"

	deep "github.com/Maxime2/go-deep"
	"github.com/theothertomelliott/acyclic"
)

// Trainer is a neural network trainer
type Trainer interface {
	Train(n *deep.Neural, examples, validation Examples, iterations int)
	SetPrefix(prefix string)
}

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	solver    Solver
	printer   *StatsPrinter
	verbosity int
}

// NewTrainer creates a new trainer
func NewTrainer(solver Solver, precision, verbosity int) *OnlineTrainer {
	if precision == 0 {
		precision = 4
	}
	return &OnlineTrainer{
		solver:    solver,
		printer:   NewStatsPrinter(precision),
		verbosity: verbosity,
	}
}

type internal struct {
	E, E_1 [][]deep.Deepfloat64
	//deltas [][]deep.Deepfloat64
	D_E_y [][]deep.Deepfloat64
	D_E_x [][]deep.Deepfloat64
}

func newE(layers []*deep.Layer) [][]deep.Deepfloat64 {
	E := make([][]deep.Deepfloat64, len(layers))
	for i, l := range layers {
		E[i] = make([]deep.Deepfloat64, len(l.Neurons))
	}
	return E
}

func newTraining(layers []*deep.Layer) *internal {
	//deltas := make([][]deep.Deepfloat64, len(layers))
	d_E_y := make([][]deep.Deepfloat64, len(layers))
	d_E_x := make([][]deep.Deepfloat64, len(layers))
	for i, l := range layers {
		//deltas[i] = make([]deep.Deepfloat64, len(l.Neurons)+1)
		d_E_y[i] = make([]deep.Deepfloat64, len(l.Neurons))
		d_E_x[i] = make([]deep.Deepfloat64, len(l.Neurons))
	}
	return &internal{
		//deltas: deltas,
		D_E_y: d_E_y,
		D_E_x: d_E_x,
		E:     newE(layers),
		E_1:   newE(layers),
	}
}

// Set new output prtefix
func (t *OnlineTrainer) SetPrefix(prefix string) {
	t.printer.SetPrefix(prefix)
}

// Train trains n
func (t *OnlineTrainer) Train(n *deep.Neural, examples, validation Examples, iterations uint32) {
	t.internal = newTraining(n.Layers)

	//train := make(Examples, len(examples))
	//copy(train, examples)

	t.printer.Init(n)
	numWeights := n.NumWeights()
	t.solver.Init(n.Layers)

	ts := time.Now()
	for i := uint32(1); i <= iterations; i++ {
		var completed int
		examples.Shuffle()
		//t.solver.InitGradients()
		t.E_1 = t.E
		t.E = newE(n.Layers)
		for j := 0; j < len(examples); j++ {
			completed = t.learn(n, examples[j], i)
		}
		for e := range t.E {
			for _, x := range t.E[e] {
				x /= deep.Deepfloat64(len(examples))
			}
		}
		//t.E /= deep.Deepfloat64(n.NumWeights()) * deep.Deepfloat64(len(examples))
		//completed = t.adjust(n, i)
		if t.verbosity > 0 && i%uint32(t.verbosity) == 0 && len(validation) > 0 {
			rCompleted := float64(completed) / float64(numWeights) * 100.0
			n.TotalError = deep.TotalError(t.E[len(n.Layers)-1])
			t.printer.PrintProgress(n, validation, time.Since(ts), i, rCompleted)
		}
		n.Config.Epoch++
		t.epoch(n, i)
		if completed == numWeights {
			break
		}
	}
	n.TotalError = deep.TotalError(t.E[len(n.Layers)-1])
}

func (t *OnlineTrainer) learn(n *deep.Neural, e Example, it uint32) int {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	return t.update(n, it)
}

func (t *OnlineTrainer) calculateDeltas(n *deep.Neural, ideal []deep.Deepfloat64) {
	loss := deep.GetLoss(n.Config.Loss)
	t.solver.ResetLr()
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.E[len(n.Layers)-1][i] += loss.F(neuron.Value, ideal[i])
		neuron.Desired = ideal[i]
		neuron.Ideal = neuron.A.If(ideal[i])
		neuron.Ln = deep.Deepfloat64(math.Log(float64((1 - neuron.Desired) / neuron.Desired)))
		//fmt.Printf(" oo i:%v; ideal: %v; neuron.Value: %v; neuron.Ideal: %v; neuron.Sum: %v\n", i, ideal[i], neuron.Value, neuron.Ideal, neuron.Sum)
		//y := neuron.DActivate(neuron.Value)
		//t.deltas[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
		//	neuron.Value,
		//	ideal[i]) * y
		t.D_E_y[len(n.Layers)-1][i] = loss.Df(neuron.Value, ideal[i])
		t.D_E_x[len(n.Layers)-1][i] = t.D_E_y[len(n.Layers)-1][i] * neuron.DActivate(neuron.Value)
		//fmt.Printf("    i:%v; dE_y: %v; dE_x: %v -- neuron.DActivate: %v\n", i, t.D_E_y[len(n.Layers)-1][i], t.D_E_x[len(n.Layers)-1][i], neuron.DActivate(neuron.Value))

		var den deep.Deepfloat64 = 0

		for _, synapse := range neuron.In {
			den += synapse.FireDelta(t.D_E_x[len(n.Layers)-1][i])
		}
		lr := float64((neuron.Ln + neuron.Sum) / den)

		t.solver.SetLr(len(n.Layers)-1, i, lr)
	}
	bottom := 0
	if n.Config.Type == deep.KolmogorovType {
		bottom = 1
	}

	for i := len(n.Layers) - 2; i >= bottom; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			//var sum deep.Deepfloat64
			var sum_y deep.Deepfloat64
			var n_ideal deep.Deepfloat64
			for k, s := range neuron.Out {
				//fmt.Printf("\t oo i:%v; j:%v; k:%v; upIdeal:%v; upSum:%v; s.Out:%v;  s.In: %v\n", i, j, k, s.Up.Ideal, s.Up.Sum, s.Out, s.In)

				gap := (s.GetUp().Ideal - s.GetUp().Sum) / deep.Deepfloat64(len(s.GetUp().In))
				n_ideal += (gap + s.GetOut() - s.GetWeight(0)) / s.GetWeight(1)
				//fmt.Printf("\t\tcnt:%v; gap: %v == n_ideal: %v; s.Up.Ideal: %v; s.Weights[0]: %v; s.Weights[1]: %v;  gap: %v\n",
				//	cnt, gap, n_ideal, s.Up.Ideal, s.Weights[0], s.Weights[1], gap)

				fd := s.FireDerivative()
				//sum += fd * t.deltas[i+1][k]
				sum_y += fd * t.D_E_x[i+1][k]
			}
			//sum *= neuron.DActivate(neuron.Value)
			//if !math.IsNaN(float64(sum)) {
			//	t.deltas[i][j] = sum
			//}
			//fmt.Printf("\t ** i:%v; j:%v; n_ideal: %v\n", i, j, n_ideal)
			n_ideal = n_ideal / deep.Deepfloat64(len(neuron.Out))
			//fmt.Printf("\t ** i:%v; j:%v; n_ideal: %v\n", i, j, n_ideal)
			n_ideal = neuron.A.Idomain(neuron.Value, n_ideal)
			//fmt.Printf("\t ** i:%v; j:%v; n_ideal: %v - Idomain; value: %v\n", i, j, n_ideal, neuron.Value)
			t.E[i][j] += loss.F(neuron.Value, n_ideal)
			neuron.Ideal = neuron.A.If(n_ideal)
			neuron.Desired = n_ideal
			neuron.Ln = deep.Deepfloat64(math.Log(float64((1 - neuron.Desired) / neuron.Desired)))
			//t.D_E_y[i][j] = loss.Df(neuron.Value, n_ideal)
			//t.D_E_x[i][j] = t.D_E_y[i][j] * neuron.DActivate(neuron.Value)

			//fmt.Printf("\t __ i:%v; j:%v; n.Value: %v; n_ideal: %v; E: %v; neuron.Ideal: %v\n", i, j, neuron.Value, n_ideal, t.E[i][j], neuron.Ideal)
			if !math.IsNaN(float64(sum_y)) {
				t.D_E_y[i][j] = sum_y
				t.D_E_x[i][j] = t.D_E_y[i][j] * neuron.DActivate(neuron.Value)
			} else {
				t.D_E_y[i][j], t.D_E_x[i][j] = 0, 0
			}

			var den deep.Deepfloat64

			for _, synapse := range neuron.In {
				den += synapse.FireDelta(t.D_E_x[i][j])
			}
			lr := float64((neuron.Ln + neuron.Sum) / den)

			t.solver.SetLr(i, j, lr)
		}
	}
	t.solver.ConcludeLr()
}

func (t *OnlineTrainer) update(neural *deep.Neural, it uint32) int {
	if neural.Config.TrainerMode == deep.UpdateTopDown {
		return t.update2(neural, it)
	}
	return t.update0(neural, it)
}

// Update from top down
func (t *OnlineTrainer) update2(neural *deep.Neural, it uint32) int {
	var completed int
	var update deep.Deepfloat64
	bottom := 0
	if neural.Config.Type == deep.KolmogorovType {
		bottom = 1
	}
	for i := len(neural.Layers) - 1; i >= bottom; i-- {
		l := neural.Layers[i]

		for j, n := range l.Neurons {
			switch l.A {
			case deep.ActivationTabulated:
				n.A.AddPoint(n.Sum, n.Desired, it, 1)
				if n.Value != n.Desired {
					fmt.Printf("* it: %v, L: %v; N: %v; -- Sum: %v; -- Value: %v; Desired: %v;\n",
						it, i, j, n.Sum, n.Value, n.Desired)
				}
				fallthrough
			default:
				for s, synapse := range n.In {
					switch l.S {
					case deep.SynapseTypeTabulated:
						synapse.AddPoint(synapse.GetIn(), n.Ideal/deep.Deepfloat64(len(n.In)), it, 1)
						if j == 0 && s == 0 && n.Value != n.Desired {
							fmt.Printf("* it: %v, L: %v; N: %v; -- Sum: %v; -- Ideal: %v; Out: %v; -- In: %v\n",
								it, i, j, n.Sum, n.Ideal, synapse.GetOut(), synapse.GetIn())
						}
					case deep.SynapseTypeAnalytic:
						for k := 0; k < synapse.Len(); k++ {
							gradient := synapse.GetGradient(t.D_E_x[i][j], k)

							delta := t.solver.Adjust(n, synapse, i, j, s, k, gradient, it)

							update = (synapse.GetWeight(k) + delta)

							if !math.IsNaN(float64(update)) && !math.IsInf(float64(update), 0) {
								synapse.SetWeight(k, update)
								// re-fire synapse with updated weights
								//synapse.Refire()
							}

						}
					}
				}
			}
		}
	}
	return completed
}

// Set epoch for Tabulated Activations
func (t *OnlineTrainer) epoch(neural *deep.Neural, epoch uint32) {
	if neural.Config.Type != deep.KolmogorovType {
		return
	}
	for i := len(neural.Layers) - 1; i >= 1; i-- {
		l := neural.Layers[i]
		for _, n := range l.Neurons {
			if l.A == deep.ActivationTabulated {
				n.A.Epoch(epoch)
			}
			if l.S == deep.SynapseTypeTabulated {
				for _, s := range n.In {
					s.Epoch(epoch)
				}
			}
		}
	}

}

// Update from bootom up
func (t *OnlineTrainer) update0(neural *deep.Neural, it uint32) int {
	var completed int
	var update deep.Deepfloat64
	for i, l := range neural.Layers {
		if neural.Config.Type == deep.KolmogorovType && i == 0 {
			continue
		}
		for j, n := range l.Neurons {
			switch l.A {
			case deep.ActivationTabulated:
				n.A.AddPoint(n.Sum, n.Desired, it, 1)
				fallthrough
			default:
				for s, synapse := range n.In {
					for k := 0; k < synapse.Len(); k++ {
						switch l.S {
						case deep.SynapseTypeTabulated:
							synapse.AddPoint(synapse.GetIn(), n.Ideal/deep.Deepfloat64(len(n.In)), it, 1)
						case deep.SynapseTypeAnalytic:
							gradient := synapse.GetGradient(t.D_E_x[i][j], k)

							delta := t.solver.Adjust(n, synapse, i, j, s, k, gradient, it)

							update = (synapse.GetWeight(k) + delta)

							if !math.IsNaN(float64(update)) && !math.IsInf(float64(update), 0) {
								//synapse.Weights_1[k] = synapse.Weights[k]
								synapse.SetWeight(k, update)
								// re-fire synapse with updated weights
								//synapse.Refire()
							}
						}
					}
				}
			}
		}
		//l.Refire()
	}
	return completed
}

// Save() saves internal of the trainer in readable JSON into file specified
func (t *OnlineTrainer) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	acyclic.Fprint(f, t.internal)
	return nil
}

func (t *OnlineTrainer) SolverSave(path string) error {
	return t.solver.Save(path)
}
