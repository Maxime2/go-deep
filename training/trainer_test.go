package training

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	deep "github.com/Maxime2/go-deep"
	"github.com/stretchr/testify/assert"
)

func Test_BoundedRegression(t *testing.T) {
	rand.Seed(0)

	funcs := []func(deep.Deepfloat64) deep.Deepfloat64{
		func(x deep.Deepfloat64) deep.Deepfloat64 { return deep.Deepfloat64(math.Sin(float64(x))) },
		func(x deep.Deepfloat64) deep.Deepfloat64 { return deep.Deepfloat64(math.Pow(float64(x), 2)) },
		func(x deep.Deepfloat64) deep.Deepfloat64 { return deep.Deepfloat64(math.Sqrt(float64(x))) },
	}

	for z, f := range funcs {

		data := Examples{}
		for i := 0.0; i < 1; i += 0.01 {
			data = append(data, Example{Input: []deep.Deepfloat64{deep.Deepfloat64(i)}, Response: []deep.Deepfloat64{f(deep.Deepfloat64(i))}})
		}
		n := deep.NewNeural(&deep.Config{
			Inputs:     1,
			Layout:     []int{4, 4, 1},
			Activation: deep.ActivationSigmoid,
			Mode:       deep.ModeRegression,
			Weight:     deep.WeightUniform,
		})

		trainer := NewTrainer(NewSGD(0.25), n.Config.LossPrecision, 100)
		trainer.Train(n, data, nil, 5000)

		tests := []deep.Deepfloat64{0.0, 0.1, 0.25, 0.5, 0.75, 0.9}
		for _, x := range tests {
			predict := float64(n.Predict([]deep.Deepfloat64{deep.Deepfloat64(x)})[0])
			assert.InEpsilon(t, float64(f(x)+1), predict+1, 0.1, "Response: %v; Predict: %v | %v; %v", f(x), predict, x, z)
		}
	}
}

func Test_RegressionLinearOuts(t *testing.T) {
	rand.Seed(0)
	squares := Examples{}
	for i := 0.0; i < 100.0; i++ {
		squares = append(squares, Example{Input: []deep.Deepfloat64{deep.Deepfloat64(i)}, Response: []deep.Deepfloat64{deep.Deepfloat64(math.Sqrt(1 + i))}})
	}
	squares.Shuffle()
	n := deep.NewNeural(&deep.Config{
		Inputs: 1,
		Layout: []int{3, 3, 1},
		//Activation: deep.ActivationReLU,
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeRegression,
		Weight:     deep.WeightNormal,
	})

	//	trainer := NewBatchTrainer(NewAdam(0.01, 0, 0, 0), 0, 25, 2)
	//rainer := NewTrainer(NewAdam(0.01, 0, 0, 0), 0)
	trainer := NewTrainer(NewSGD(0.01), n.Config.LossPrecision, 10000)
	trainer.Train(n, squares, squares, 250000)

	for i := 0; i < 100; i++ {
		x := deep.Deepfloat64(rand.Intn(99) + 1)
		assert.InEpsilon(t, math.Sqrt(float64(1+x))+1, float64(n.Predict([]deep.Deepfloat64{x})[0]+1), 0.1, "for %+v want: %+v have: %+v\n", x, math.Sqrt(float64(x))+1, float64(n.Predict([]deep.Deepfloat64{x})[0]+1))
	}
}

func Test_Training(t *testing.T) {
	rand.Seed(0)

	data := Examples{
		Example{[]deep.Deepfloat64{0}, []deep.Deepfloat64{0}},
		Example{[]deep.Deepfloat64{0}, []deep.Deepfloat64{0}},
		Example{[]deep.Deepfloat64{0}, []deep.Deepfloat64{0}},
		Example{[]deep.Deepfloat64{5}, []deep.Deepfloat64{1}},
		Example{[]deep.Deepfloat64{5}, []deep.Deepfloat64{1}},
	}

	n := deep.NewNeural(&deep.Config{
		Inputs:     1,
		Layout:     []int{5, 1},
		Activation: deep.ActivationSigmoid,
		Weight:     deep.WeightUniform,
	})

	trainer := NewTrainer(NewSGD(0.5), n.Config.LossPrecision, 0)
	trainer.Train(n, data, nil, 1000)

	v := n.Predict([]deep.Deepfloat64{0})
	assert.InEpsilon(t, 1, float64(1+v[0]), 0.1)
	v = n.Predict([]deep.Deepfloat64{5})
	assert.InEpsilon(t, 1.0, float64(v[0]), 0.1)
}

var data = []Example{
	{[]deep.Deepfloat64{2.7810836, 2.550537003}, []deep.Deepfloat64{0}},
	{[]deep.Deepfloat64{1.465489372, 2.362125076}, []deep.Deepfloat64{0}},
	{[]deep.Deepfloat64{3.396561688, 4.400293529}, []deep.Deepfloat64{0}},
	{[]deep.Deepfloat64{1.38807019, 1.850220317}, []deep.Deepfloat64{0}},
	{[]deep.Deepfloat64{3.06407232, 3.005305973}, []deep.Deepfloat64{0}},
	{[]deep.Deepfloat64{7.627531214, 2.759262235}, []deep.Deepfloat64{1}},
	{[]deep.Deepfloat64{5.332441248, 2.088626775}, []deep.Deepfloat64{1}},
	{[]deep.Deepfloat64{6.922596716, 1.77106367}, []deep.Deepfloat64{1}},
	{[]deep.Deepfloat64{8.675418651, -0.242068655}, []deep.Deepfloat64{1}},
	{[]deep.Deepfloat64{7.673756466, 3.508563011}, []deep.Deepfloat64{1}},
}

func Test_Prediction(t *testing.T) {
	rand.Seed(0)

	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{2, 2, 1},
		Activation: deep.ActivationSigmoid,
		Weight:     deep.WeightUniform,
	})
	trainer := NewTrainer(NewSGD(0.5), n.Config.LossPrecision, 0)

	trainer.Train(n, data, nil, 5000)

	for _, d := range data {
		assert.InEpsilon(t, float64(n.Predict(d.Input)[0]+1), float64(d.Response[0]+1), 0.1)
	}
}

func Test_CrossVal(t *testing.T) {
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{1, 1},
		Activation: deep.ActivationTanh,
		Loss:       deep.LossMeanSquared,
		Weight:     deep.WeightUniform,
	})

	trainer := NewTrainer(NewSGD(0.5), n.Config.LossPrecision, 0)
	trainer.Train(n, data, data, 1000)

	for _, d := range data {
		assert.InEpsilon(t, float64(n.Predict(d.Input)[0]+1), float64(d.Response[0]+1), 0.1)
		assert.InEpsilon(t, 1, float64(crossValidate(n, data)+1), 0.01)
	}
}

func Test_MultiClass(t *testing.T) {
	var data = []Example{
		{[]deep.Deepfloat64{2.7810836, 2.550537003}, []deep.Deepfloat64{1, 0}},
		{[]deep.Deepfloat64{1.465489372, 2.362125076}, []deep.Deepfloat64{1, 0}},
		{[]deep.Deepfloat64{3.396561688, 4.400293529}, []deep.Deepfloat64{1, 0}},
		{[]deep.Deepfloat64{1.38807019, 1.850220317}, []deep.Deepfloat64{1, 0}},
		{[]deep.Deepfloat64{3.06407232, 3.005305973}, []deep.Deepfloat64{1, 0}},
		{[]deep.Deepfloat64{7.627531214, 2.759262235}, []deep.Deepfloat64{0, 1}},
		{[]deep.Deepfloat64{5.332441248, 2.088626775}, []deep.Deepfloat64{0, 1}},
		{[]deep.Deepfloat64{6.922596716, 1.77106367}, []deep.Deepfloat64{0, 1}},
		{[]deep.Deepfloat64{8.675418651, -0.242068655}, []deep.Deepfloat64{0, 1}},
		{[]deep.Deepfloat64{7.673756466, 3.508563011}, []deep.Deepfloat64{0, 1}},
	}

	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{2, 2},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Loss:       deep.LossMeanSquared,
		Weight:     deep.WeightUniform,
	})

	trainer := NewTrainer(NewSGD(0.01), n.Config.LossPrecision, 100)
	trainer.Train(n, data, data, 2000)

	for _, d := range data {
		est := n.Predict(d.Input)
		assert.InEpsilon(t, 1.0, float64(deep.Sum(est)), 0.00001)
		if d.Response[0] == 1.0 {
			assert.InEpsilon(t, float64(n.Predict(d.Input)[0]+1), float64(d.Response[0]+1), 0.1)
		} else {
			assert.InEpsilon(t, float64(n.Predict(d.Input)[1]+1), float64(d.Response[1]+1), 0.1)
		}
		assert.InEpsilon(t, 1, float64(crossValidate(n, data)+1), 0.01)
	}

}

func Test_or(t *testing.T) {
	rand.Seed(0)
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{1, 1},
		Activation: deep.ActivationTanh,
		Mode:       deep.ModeBinary,
		Weight:     deep.WeightUniform,
	})
	permutations := Examples{
		{[]deep.Deepfloat64{0, 0}, []deep.Deepfloat64{0}},
		{[]deep.Deepfloat64{1, 0}, []deep.Deepfloat64{1}},
		{[]deep.Deepfloat64{0, 1}, []deep.Deepfloat64{1}},
		{[]deep.Deepfloat64{1, 1}, []deep.Deepfloat64{1}},
	}

	trainer := NewTrainer(NewSGD(0.5), n.Config.LossPrecision, 10)

	trainer.Train(n, permutations, permutations, 25)

	for _, perm := range permutations {
		assert.Equal(t, deep.Round(n.Predict(perm.Input)[0]), perm.Response[0])
	}
}

func Test_xor(t *testing.T) {
	rand.Seed(0)
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{3, 1}, // Sufficient for modeling (AND+OR) - with 5-6 neuron always converges
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.WeightUniform,
	})
	permutations := Examples{
		{[]deep.Deepfloat64{0, 0}, []deep.Deepfloat64{0}},
		{[]deep.Deepfloat64{1, 0}, []deep.Deepfloat64{1}},
		{[]deep.Deepfloat64{0, 1}, []deep.Deepfloat64{1}},
		{[]deep.Deepfloat64{1, 1}, []deep.Deepfloat64{0}},
	}

	trainer := NewTrainer(NewSGD(1.0), n.Config.LossPrecision, 50)
	trainer.Train(n, permutations, permutations, 500)

	for _, perm := range permutations {
		assert.InEpsilon(t, float64(n.Predict(perm.Input)[0]+1), float64(perm.Response[0]+1), 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, n.Predict(perm.Input)[0]+1, perm.Response[0]+1)
	}
}

func Test_essential(t *testing.T) {
	rand.Seed(0)
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{5, 1}, // Sufficient for modeling (AND+OR) - with 5-6 neuron always converges
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.WeightUniform,
		Degree:     1,
	})
	permutations := Examples{
		{[]deep.Deepfloat64{0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1}, []deep.Deepfloat64{0.5}},
		{[]deep.Deepfloat64{0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5}, []deep.Deepfloat64{0.5}},
	}

	trainer := NewTrainer(NewSGD(0.01), n.Config.LossPrecision, 5000)
	trainer.SetPrefix("essential ")
	trainer.Train(n, permutations, permutations, 50000)

	n.Dot("essential-test.dot")
	n.InputStats(true, "essential-test.stats")
	n.SaveReadable("essential-test.neural")
	trainer.SolverSave("essential-test.sgd")
	trainer.Save("essential-test.trainer")

	for _, perm := range permutations {
		assert.InEpsilon(t, float64(n.Predict(perm.Input)[0]+1), float64(perm.Response[0]+1), 0.2, "input: %+v; want: %+v have: %+v\n", perm.Input, n.Predict(perm.Input)[0]+1, perm.Response[0]+1)
	}
}

func printResult(ideal, actual []float64) {
	fmt.Printf("want: %+v have: %+v\n", ideal, actual)
}

func Test_RHW(t *testing.T) {
	c := deep.Config{
		Degree:        3,
		Inputs:        6,
		Layout:        []int{36, 2*6 + 1, 1},
		Activation:    deep.ActivationSigmoid,
		Mode:          deep.ModeBinary,
		LossPrecision: 12,
		Weight:        deep.WeightUniform,
	}
	n := deep.NewNeural(&c)
	nn, _ := deep.Load("rhw-test.init")
	permutations := Examples{
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.1, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.5, 0.1, 0.1}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.1, 0.5, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.1, 0.5, 0.1}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.1, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.5, 0.5, 0.1}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.1, 0.5, 0.5, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},

		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.1, 0.1, 0.5}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.1, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.5, 0.1, 0.5}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.1, 0.5, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.1, 0.5, 0.5}, []deep.Deepfloat64{.5}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.1, 0.5, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.1, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.1, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.1, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.1, 0.5, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.5, 0.1, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.5, 0.1, 0.5}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.5, 0.5, 0.1}, []deep.Deepfloat64{0.1}},
		{[]deep.Deepfloat64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, []deep.Deepfloat64{.5}},
	}

	for i, p := range permutations {
		predict := float64(nn.Predict(p.Input)[0])
		name := fmt.Sprintf("rhw-test-post-%d.neural", i)
		nn.SaveReadable(name)
		assert.InEpsilon(t, 1+float64(p.Response[0]), 1+predict, 0.0005, "Response: %v; Predict: %v | %v", p.Response[0], predict, p.Input)
	}

	nn.Dot("rhw-test-0.dot")
	//nn.SaveReadable("rhw-test-post-0.neural")

	n.SaveReadable("rhw-test-pre.neural")
	n.Save("rhw-test.dump")
	trainer := NewTrainer(NewSGD(0.1), n.Config.LossPrecision, 1)
	trainer.SetPrefix("RHW ")
	trainer.Train(n, permutations, permutations, 1425)
	trainer.SolverSave("rhw-test.sgd")
	trainer.Save("rhw-test.trainer")

	n.Dot("rhw-test.dot")
	n.InputStats(true, "rhw-test.stats")
	for _, p := range permutations {
		predict := float64(n.Predict(p.Input)[0])
		assert.InEpsilon(t, 1+float64(p.Response[0]), 1+predict, 0.0005, "Response: %v; Predict: %v | %v", p.Response[0], predict, p.Input)
	}
	n.SaveReadable("rhw-test-post.neural")

	x := 700.0
	for i := 0; i < 20; i++ {
		r := 1 / (1 + math.Exp(x))
		fmt.Printf(" oo %v :: %v | %v | %v\n", x, r, 1-r, r*(1-r))
		x += 1.0
	}
	fmt.Printf(" Of -Inf: %v\n", 1/(1+math.Inf(-1)))

	activation := deep.GetActivation(deep.ActivationSigmoid)
	x = -300.0
	for i := 0; i < 20; i++ {
		y := math.Pow(10, x-float64(i))
		r := activation.If(deep.Deepfloat64(y))
		fmt.Printf(" oo %v :: %v\n", y, r)
	}

	for x := -10; x > -24; x-- {
		y := math.Pow(10, float64(x))
		fmt.Printf(" dd %v :: %v  %v\n", x, y, 1.0-y)
	}
}
