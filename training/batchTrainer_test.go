package training

import (
	"math/rand"
	"runtime"
	"testing"

	deep "github.com/Maxime2/go-deep"
)

func Benchmark_xor(b *testing.B) {
	rand.Seed(0)
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{32, 32, 1},
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.WeightUniform,
	})
	exs := Examples{
		{[]deep.Deepfloat64{0, 0}, []deep.Deepfloat64{0}},
		{[]deep.Deepfloat64{1, 0}, []deep.Deepfloat64{1}},
		{[]deep.Deepfloat64{0, 1}, []deep.Deepfloat64{1}},
		{[]deep.Deepfloat64{1, 1}, []deep.Deepfloat64{0}},
	}
	const minExamples = 4000
	var dupExs Examples
	for len(dupExs) < minExamples {
		dupExs = append(dupExs, exs...)
	}

	for i := 0; i < b.N; i++ {
		const iterations = 20
		//solver := NewAdam(0.001, 0.9, 0.999, 1e-8)
		solver := NewSGD(0.001)
		trainer := NewBatchTrainer(solver, n.Config.LossPrecision, iterations, len(dupExs)/2, runtime.NumCPU())
		trainer.Train(n, dupExs, dupExs, iterations)
	}
}
