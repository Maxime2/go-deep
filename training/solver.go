package training

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"math"
	"os"

	deep "github.com/Maxime2/go-deep"
)

// Solver implements an update rule for training a NN
type Solver interface {
	Init(size int)
	Update(value, gradient, in deep.Deepfloat64, iteration, idx int)
	Adjust(synapse *deep.Synapse, k, iteration, idx int) (deep.Deepfloat64, bool)
	InitGradients()
	Save(path string) error
	Load(path string) error
	Gradient(idx int) float64
}

// SGD is stochastic gradient descent with nesterov/momentum
type SGD struct {
	Lr       float64
	Decay    float64
	Momentum float64
	//nesterov bool
	Moments     []deep.Deepfloat64
	Lrs         []deep.Deepfloat64
	Gradients   []deep.Deepfloat64
	Gradients_1 []deep.Deepfloat64
}

// NewSGD returns a new SGD solver
func NewSGD(lr, momentum, decay float64, nesterov bool) *SGD {
	return &SGD{
		Lr:       fparam(lr, 0.01),
		Decay:    decay,
		Momentum: momentum,
		//nesterov: nesterov,
	}
}

// Init initializes vectors using number of weights in network
func (o *SGD) Init(size int) {
	o.Moments = make([]deep.Deepfloat64, size)
	o.Gradients_1 = make([]deep.Deepfloat64, size)
	o.Lrs = make([]deep.Deepfloat64, size)
	for i := 0; i < size; i++ {
		o.Lrs[i] = deep.Deepfloat64(o.Lr)
	}
}

// Initialise Gradients
func (o *SGD) InitGradients() {
	o.Gradients = make([]deep.Deepfloat64, len(o.Moments))
}

// Update updates cumulative gradient for a given weight
func (o *SGD) Update(value, gradient, in deep.Deepfloat64, iteration, idx int) {
	o.Gradients[idx] += gradient
}

// Adjust returns the update for a given weight and adjusts learnig rate based on gradint signs
func (o *SGD) Adjust(synapse *deep.Synapse, k, iteration, idx int) (deep.Deepfloat64, bool) {
	var newValue deep.Deepfloat64
	fakeRoot := false

	if iteration > 3 {
		value := synapse.Weights[k]
		value_1 := synapse.Weights_1[k]
		fx := synapse.WeightFunction(o.Gradients[idx], k)
		fx_1 := synapse.WeightFunction(o.Gradients_1[idx], k)
		d_inv := (value - value_1) / (fx - fx_1)

		newValue = deep.Deepfloat64(o.Momentum)*o.Moments[idx] - d_inv*fx
		if math.IsNaN(float64(newValue)) {
			newValue = deep.Deepfloat64(o.Momentum)*o.Moments[idx] - o.Lrs[idx]*o.Gradients[idx]
		} else {
			fakeRoot = math.Signbit(float64(d_inv))
			//o.Gradients[idx] = fx
		}
	} else {
		newValue = deep.Deepfloat64(o.Momentum)*o.Moments[idx] - o.Lrs[idx]*o.Gradients[idx]
	}
	if !math.IsNaN(float64(newValue)) {
		o.Moments[idx] = newValue
	}

	if math.Signbit(float64(o.Gradients[idx])) != math.Signbit(float64(o.Gradients_1[idx])) {
		if o.Lrs[idx] > deep.Eps {
			o.Lrs[idx] *= 0.95
		}
	} else {
		if o.Lrs[idx] < deep.Deepfloat64(o.Lr) {
			o.Lrs[idx] *= 1 / 0.95
		}
	}
	if o.Lrs[idx] < deep.Eps {
		o.Lrs[idx] = deep.Eps
	}
	o.Gradients_1[idx] = o.Gradients[idx]

	return o.Moments[idx], fakeRoot
}

// Gradient returns gradient value for an index
func (o *SGD) Gradient(idx int) float64 {
	return float64(o.Gradients[idx])
}

// Save saves SGD into the file specified to be loaded later
func (o *SGD) Save(path string) error {
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
func (o *Adam) Init(size int) {
	o.v, o.m = make([]float64, size), make([]float64, size)
	o.gradints = make([]float64, size)
}

// Initialise gradients
func (o *Adam) InitGradients() {

}

// Adjust learning rates based on gradient signs
func (o *Adam) Adjust(synapse *deep.Synapse, k, t, idx int) (deep.Deepfloat64, bool) {
	//value := synapse.Weights[k]
	gradient := o.gradints[idx]

	lrt := deep.Deepfloat64(o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, float64(t)))) /
		(1.0 - math.Pow(o.beta, float64(t))))
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
