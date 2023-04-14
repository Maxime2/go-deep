package training

import (
	"math"
	"os"
	"time"

	deep "github.com/Maxime2/go-deep"
	"github.com/theothertomelliott/acyclic"
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
		t.E_1 = t.E
		t.E = newE(n.Layers)
		n.Config.N_iterations = deep.MinIterations
		for j := 0; j < len(examples); j++ {
			t.learn(n, examples[j], i)
		}
		for i := range t.E {
			for _, x := range t.E[i] {
				x /= deep.Deepfloat64(len(examples))
			}
		}
		//t.E /= deep.Deepfloat64(n.NumWeights()) * deep.Deepfloat64(len(examples))
		completed := t.adjust(n, i)
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			rCompleted := float64(completed) / float64(numWeights) * 100.0
			t.printer.PrintProgress(n, t.E[len(n.Layers)-1], validation, time.Since(ts), i, rCompleted)
		}
		if completed == numWeights {
			break
		}
	}
}

func (t *OnlineTrainer) learn(n *deep.Neural, e Example, it int) {
	n.Forward(e.Input)
	t.calculateDeltas(n, e.Response)
	t.update(n, it)
}

func (t *OnlineTrainer) calculateDeltas(n *deep.Neural, ideal []deep.Deepfloat64) {
	loss := deep.GetLoss(n.Config.Loss)
	activation := deep.GetActivation(n.Layers[len(n.Layers)-1].A)
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.E[len(n.Layers)-1][i] += loss.F(neuron.Value, ideal[i])
		neuron.Ideal = activation.If(ideal[i])
		//fmt.Printf(" oo i:%v; ideal: %v; neuron.ideal: %v\n", i, ideal[i], neuron.Ideal)
		y := neuron.DActivate(neuron.Value)
		//t.deltas[len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
		//	neuron.Value,
		//	ideal[i]) * y
		t.D_E_y[len(n.Layers)-1][i] = loss.Df(
			neuron.Value,
			ideal[i])
		t.D_E_x[len(n.Layers)-1][i] = t.D_E_y[len(n.Layers)-1][i] * y
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		activation = deep.GetActivation(n.Layers[i].A)
		for j, neuron := range n.Layers[i].Neurons {
			//var sum deep.Deepfloat64
			//var sum_y deep.Deepfloat64
			var n_ideal deep.Deepfloat64
			for _ /*k*/, s := range neuron.Out {
				//fmt.Printf("\t oo i:%v; j:%v; k:%v;; upIdeal:%v; upSum:%v; s.Out:%v;  s.In: %v\n", i, j, k, s.Up.Ideal, s.Up.Sum, s.Out, s.In)
				if math.Signbit(float64(s.Up.Ideal)) != math.Signbit(float64(s.Out)) {
					n_ideal += s.Up.Ideal * s.Up.Sum
				} else {
					n_ideal += s.Up.Ideal / s.Up.Sum
				}
				//fd := s.FireDerivative()
				//sum += fd * t.deltas[i+1][k]
				//sum_y += fd * t.D_E_x[i+1][k]
			}
			//sum *= neuron.DActivate(neuron.Value)
			//if !math.IsNaN(float64(sum)) {
			//	t.deltas[i][j] = sum
			//}
			//fmt.Printf("\t ** i:%v; j:%v; n_ideal: %v\n", i, j, n_ideal)
			n_ideal = activation.Idomain(neuron.Value * n_ideal * deep.Deepfloat64(len(neuron.Out)))
			t.E[i][j] += loss.F(neuron.Value, n_ideal)
			neuron.Ideal = activation.If(n_ideal)
			t.D_E_y[i][j] = loss.Df(neuron.Value, n_ideal)
			t.D_E_x[i][j] = t.D_E_y[i][j] * neuron.DActivate(neuron.Value)

			//fmt.Printf(" __ i:%v; j:%v; ideal: %v; neuron.ideal: %v\n", i, j, n_ideal, neuron.Ideal)
			//if !math.IsNaN(float64(sum_y)) {
			//	t.D_E_y[i][j] = sum_y
			//	t.D_E_x[i][j] = t.D_E_y[i][j] * neuron.DActivate(neuron.Value)
			//}
		}
	}
}

func (t *OnlineTrainer) update(n *deep.Neural, it int) {
	var idx int
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for _, synapse := range l.Neurons[j].In {
				for k := 0; k < len(synapse.Weights); k++ {
					gradient := synapse.GetGradient(t.D_E_x[i][j], k)
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
	var update deep.Deepfloat64
	minLayerFakeRoot := len(n.Layers)
	for i, l := range n.Layers {
		for j, neuron := range l.Neurons {
			for _, synapse := range l.Neurons[j].In {
				for k := 0; k < len(synapse.Weights); k++ {
					if l.Number > minLayerFakeRoot {
						synapse.ClearFakeRoots(k)
					}
					if !synapse.IsComplete[k] {
						delta, fakeRoot, itsdone := t.solver.Adjust(synapse, k, it, idx, neuron.Ideal /*t.E[i][j]*/, t.E_1[i][j])
						if itsdone {
							completed++
							synapse.IsComplete[k] = true
						} else {

							update = (synapse.Weights[k] + delta)

							//if idx >= 0 {
							//	fmt.Printf("  Trainer: update: %v\n", update)
							//}
							if !math.IsNaN(float64(update)) {
								if it > 2 {
									if (update-synapse.Weights[k])/(1-(update-synapse.Weights[k])/(synapse.Weights[k]-synapse.Weights_1[k])) < deep.Eps {
										//synapse.IsComplete[k] = true
										if fakeRoot && l.Number <= minLayerFakeRoot {
											synapse.AddFakeRoot(k, update)
											update = n.Config.Weight()
											minLayerFakeRoot = l.Number
										} else {
											//completed++
											if math.Abs(float64(update-synapse.Weights[k]))/math.Abs(float64(synapse.Weights[k]-synapse.Weights_1[k])) > 1 {
												if update > synapse.Weights[k] {
													update = deep.Deepfloat64(math.Abs(float64(synapse.Weights[k]-synapse.Weights_1[k]))) - deep.Eps + synapse.Weights[k]
												} else {
													update = synapse.Weights[k] + deep.Eps - deep.Deepfloat64(math.Abs(float64(synapse.Weights[k]-synapse.Weights_1[k])))
												}
											}
										}
									}
								}
								//if idx == 0 {
								//	fmt.Printf("  Trainer:: idx: %d; weight: %v; weight_1: %v; update: %v\n", idx, synapse.Weights[k], synapse.Weights_1[k], update)
								//}
								synapse.Weights_1[k] = synapse.Weights[k]
								synapse.Weights[k] = update
								//if idx == 0 {
								//	fmt.Printf("  Trainer:: idx: %d; weight: %v; weight_1: %v; delta: %v\n", idx, synapse.Weights[k], synapse.Weights_1[k], delta)
								//}
							}
						}
					} else {
						completed++
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
