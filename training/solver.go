package training

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"

	deep "github.com/Maxime2/go-deep"
	"github.com/theothertomelliott/acyclic"
)

// Solver implements an update rule for training a NN
type Solver interface {
	Init(layers []*deep.Layer)
	SetGradient(i, j, s, k int, gradient deep.Deepfloat64)
	Adjust(i, j, s, k int, gradient deep.Deepfloat64, iteration int) (deep.Deepfloat64, bool)
	Save(path string) error
	Load(path string) error
	//Gradient(idx int) float64
}

// SGD is stochastic gradient descent with nesterov/momentum
type SGD struct {
	Lr        float64
	Moments   [][][][]deep.Deepfloat64
	Lrs       [][][][]deep.Deepfloat64
	Gradients [][][][]deep.Deepfloat64
	//Gradients_1 []deep.Deepfloat64
}

// NewSGD returns a new SGD solver
func NewSGD(lr float64) *SGD {
	return &SGD{
		Lr: fparam(lr, 0.01),
	}
}

// Init initializes vectors using number of weights in network
func (o *SGD) Init(layers []*deep.Layer) {
	o.Moments = make([][][][]deep.Deepfloat64, len(layers))
	o.Gradients = make([][][][]deep.Deepfloat64, len(layers))
	o.Lrs = make([][][][]deep.Deepfloat64, len(layers))
	for i, l := range layers {
		o.Moments[i] = make([][][]deep.Deepfloat64, len(l.Neurons))
		o.Gradients[i] = make([][][]deep.Deepfloat64, len(l.Neurons))
		o.Lrs[i] = make([][][]deep.Deepfloat64, len(l.Neurons))
		for j, n := range l.Neurons {
			o.Moments[i][j] = make([][]deep.Deepfloat64, len(n.In))
			o.Gradients[i][j] = make([][]deep.Deepfloat64, len(n.In))
			o.Lrs[i][j] = make([][]deep.Deepfloat64, len(n.In))
			for k, synapse := range l.Neurons[j].In {
				o.Moments[i][j][k] = make([]deep.Deepfloat64, len(synapse.Weights))
				o.Gradients[i][j][k] = make([]deep.Deepfloat64, len(synapse.Weights))
				o.Lrs[i][j][k] = make([]deep.Deepfloat64, len(synapse.Weights))
				for y := 0; y < len(synapse.Weights); y++ {
					o.Lrs[i][j][k][y] = deep.Deepfloat64(o.Lr)
				}
			}
		}
	}
}

func (o *SGD) SetGradient(i, j, s, k int, gradient deep.Deepfloat64) {
	o.Gradients[i][j][s][k] = gradient
}

// Adjust returns the update for a given weight and adjusts learnig rate based on gradint signs
func (o *SGD) Adjust(i, j, s, k int, gradient deep.Deepfloat64, iteration int) (deep.Deepfloat64, bool) {
	var newValue deep.Deepfloat64
	completed := false
	fx := o.Gradients[i][j][s][k]

	if math.Signbit(float64(gradient)) != math.Signbit(float64(fx)) {
		o.Lrs[i][j][s][k] *= 0.95
	} else {
		o.Lrs[i][j][s][k] *= 1 / 0.95
	}

	if iteration > 2 {

		//if math.Abs(float64(gradient)) < deep.Eps {
		//	completed = true
		//	newValue = 0
		//} else {
			newValue = -o.Lrs[i][j][s][k] * gradient
		//}
	} else {
		newValue = -o.Lrs[i][j][s][k] * gradient
	}
	if !math.IsNaN(float64(newValue)) {
		o.Moments[i][j][s][k] = newValue
	}
	o.Gradients[i][j][s][k] = gradient

	return o.Moments[i][j][s][k], completed
}

// Gradient returns gradient value for an index
//func (o *SGD) Gradient(idx int) float64 {
//	return float64(o.Gradients[idx])
//}

// Save saves SGD into the file specified to be loaded later
func (o *SGD) Save(path string) error {

	ff, err := os.Create(fmt.Sprintf("%s.readable", path))
	if err != nil {
		return err
	}
	defer ff.Close()
	acyclic.Fprint(ff, o)

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(o)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, bytes.NewReader(b))
	return err
}

// Load retrieves SGD from the file specified created using Save method
func (o *SGD) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	b, _ := ioutil.ReadAll(f)
	if err := json.Unmarshal(b, o); err != nil {
		return err
	}
	return nil
}

// Adam is an Adam solver
type Adam struct {
	lr      float64
	beta    float64
	beta2   float64
	epsilon float64

	v, m     []float64
	gradints []float64
}

// NewAdam returns a new Adam solver
func NewAdam(lr, beta, beta2, epsilon float64) *Adam {
	return &Adam{
		lr:      fparam(lr, 0.001),
		beta:    fparam(beta, 0.9),
		beta2:   fparam(beta2, 0.999),
		epsilon: fparam(epsilon, 1e-8),
	}
}

// Init initializes vectors using number of weights in network
func (o *Adam) Init(layers []*deep.Layer) {
	size := 1000
	o.v, o.m = make([]float64, size), make([]float64, size)
	o.gradints = make([]float64, size)
}

// Adjust learning rates based on gradient signs
func (o *Adam) Adjust(i, j, s, l int, gradient deep.Deepfloat64, iteration int) (deep.Deepfloat64, bool) {
	//value := synapse.Weights[k]
	//gradient := o.gradints[i][j][s][k]
	idx := 0

	lrt := deep.Deepfloat64(o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, float64(gradient)))) /
		(1.0 - math.Pow(o.beta, float64(gradient))))
	o.m[idx] = o.beta*o.m[idx] + (1.0-o.beta)*float64(gradient)
	o.v[idx] = o.beta2*o.v[idx] + (1.0-o.beta2)*math.Pow(float64(gradient), 2.0)

	return -lrt * deep.Deepfloat64(o.m[idx]/(math.Sqrt(o.v[idx])+o.epsilon)), false
}

// Update returns the update for a given weight
func (o *Adam) Update(value, gradient, in deep.Deepfloat64, t, idx int) {
	o.gradints[idx] = float64(gradient)
}

// Initialise gradients
func (o *Adam) Gradient(idx int) float64 {
	return 0.0
}

// Save saves Adam into the file specified to be loaded later
func (o *Adam) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(o)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, bytes.NewReader(b))
	return err
}

// Load retrieves Adam from the file specified created using Save method
func (o *Adam) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	b, _ := ioutil.ReadAll(f)
	if err := json.Unmarshal(b, o); err != nil {
		return err
	}
	return nil
}

func fparam(val, fallback float64) float64 {
	if val == 0.0 {
		return fallback
	}
	return val
}

func iparam(val, fallback int) int {
	if val == 0 {
		return fallback
	}
	return val
}
