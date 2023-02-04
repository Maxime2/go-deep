package training

import (
	"math"
	"sync"
	"time"

	deep "github.com/Maxime2/go-deep"
)

// BatchTrainer implements parallelized batch training
type BatchTrainer struct {
	*internalb
	verbosity   int
	batchSize   int
	parallelism int
	solver      Solver
	printer     *StatsPrinter
}

type internalb struct {
	deltas            [][][]deep.Deepfloat64
	partialDeltas     [][][][]deep.Deepfloat64
	accumulatedDeltas [][][]deep.Deepfloat64
	moments           [][][]float64
}

func newBatchTraining(layers []*deep.Layer, parallelism int) *internalb {
	deltas := make([][][]deep.Deepfloat64, parallelism)
	partialDeltas := make([][][][]deep.Deepfloat64, parallelism)
	accumulatedDeltas := make([][][]deep.Deepfloat64, len(layers))
	for w := 0; w < parallelism; w++ {
		deltas[w] = make([][]deep.Deepfloat64, len(layers))
		partialDeltas[w] = make([][][]deep.Deepfloat64, len(layers))

		for i, l := range layers {
			deltas[w][i] = make([]deep.Deepfloat64, len(l.Neurons))
			accumulatedDeltas[i] = make([][]deep.Deepfloat64, len(l.Neurons))
			partialDeltas[w][i] = make([][]deep.Deepfloat64, len(l.Neurons))
			for j, n := range l.Neurons {
				partialDeltas[w][i][j] = make([]deep.Deepfloat64, len(n.In))
				accumulatedDeltas[i][j] = make([]deep.Deepfloat64, len(n.In))
			}
		}
	}
	return &internalb{
		deltas:            deltas,
		partialDeltas:     partialDeltas,
		accumulatedDeltas: accumulatedDeltas,
	}
}

// NewBatchTrainer returns a BatchTrainer
func NewBatchTrainer(solver Solver, verbosity, batchSize, parallelism int) *BatchTrainer {
	return &BatchTrainer{
		solver:      solver,
		verbosity:   verbosity,
		batchSize:   iparam(batchSize, 1),
		parallelism: iparam(parallelism, 1),
		printer:     NewStatsPrinter(),
	}
}

// Train trains n
func (t *BatchTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internalb = newBatchTraining(n.Layers, t.parallelism)

	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Example, t.parallelism)
	nets := make([]*deep.Neural, t.parallelism)

	wg := sync.WaitGroup{}
	for i := 0; i < t.parallelism; i++ {
		nets[i] = deep.NewNeural(n.Config)

		go func(id int, workCh <-chan Example) {
			n := nets[id]
			for e := range workCh {
				n.Forward(e.Input)
				t.calculateDeltas(n, e.Response, id)
				wg.Done()
			}
		}(i, workCh)
	}

	t.printer.Init(n)
	t.solver.Init(n.NumWeights())

	ts := time.Now()
	for it := 1; it <= iterations; it++ {
		train.Shuffle()
		batches := train.SplitSize(t.batchSize)

		for _, b := range batches {
			currentWeights := n.Weights()
			for _, n := range nets {
				n.ApplyWeights(currentWeights)
			}

			wg.Add(len(b))
			for _, item := range b {
				workCh <- item
			}
			wg.Wait()

			for _, wPD := range t.partialDeltas {
				for i, iPD := range wPD {
					iAD := t.accumulatedDeltas[i]
					for j, jPD := range iPD {
						jAD := iAD[j]
						for k, v := range jPD {
							jAD[k] += v
							jPD[k] = 0
						}
					}
				}
			}

			t.update(n, it)
		}

		if t.verbosity > 0 && it%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(n, validation, time.Since(ts), it, 0.0)
		}
	}
}

func (t *BatchTrainer) calculateDeltas(n *deep.Neural, ideal []deep.Deepfloat64, wid int) {
	loss := deep.GetLoss(n.Config.Loss)
	deltas := t.deltas[wid]
	partialDeltas := t.partialDeltas[wid]
	lastDeltas := deltas[len(n.Layers)-1]

	for i, n := range n.Layers[len(n.Layers)-1].Neurons {
		lastDeltas[i] = loss.Df(
			n.Value,
			ideal[i]) *
			n.DActivate(n.Value)
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		l := n.Layers[i]
		iD := deltas[i]
		nextD := deltas[i+1]
		for j, n := range l.Neurons {
			var sum deep.Deepfloat64
			for k, s := range n.Out {
				sum += s.FireDerivative(n.Value) * nextD[k]
			}
			sum *= n.DActivate(n.Value)
			if math.IsNaN(float64(sum)) {
				iD[j] = n.DActivate(n.Value)
			} else {
				iD[j] = n.DActivate(n.Value) * sum
			}
		}
	}

	for i, l := range n.Layers {
		iD := deltas[i]
		iPD := partialDeltas[i]
		for j, n := range l.Neurons {
			jD := iD[j]
			jPD := iPD[j]
			for k, s := range n.In {
				jPD[k] += jD * s.In
			}
		}
	}
}

func (t *BatchTrainer) update(n *deep.Neural, it int) {
	var idx int
	for i, l := range n.Layers {
		iAD := t.accumulatedDeltas[i]
		for j, n := range l.Neurons {
			jAD := iAD[j]
			for k, s := range n.In {
				update := s.Weights[1] + t.solver.Adjust(s.Weights[1],
					jAD[k],
					s.In,
					it,
					idx)
				if !math.IsNaN(float64(update)) {
					s.Weights[1] = update
				}
				jAD[k] = 0
				idx++
			}
		}
	}
}
