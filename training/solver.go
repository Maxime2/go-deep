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
	Init(size int)
	Update(value, gradient, in deep.Deepfloat64, iteration, idx int)
	Adjust(synapse *deep.Synapse, k, iteration, idx int, E, E_1 deep.Deepfloat64) (deep.Deepfloat64, bool, bool)
	InitGradients()
	Save(path string) error
	Load(path string) error
	Gradient(idx int) float64
}

// SGD is stochastic gradient descent with nesterov/momentum
type SGD struct {
	Lr          float64
	Moments     []deep.Deepfloat64
	Lrs         []deep.Deepfloat64
	Gradients   []deep.Deepfloat64
	Gradients_1 []deep.Deepfloat64
}

// NewSGD returns a new SGD solver
func NewSGD(lr float64) *SGD {
	return &SGD{
		Lr: fparam(lr, 0.01),
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
	newGradient := o.Gradients[idx] + gradient
	if math.IsInf(float64(newGradient), -1) {
		o.Gradients[idx] = 1e-99
	} else if math.IsInf(float64(newGradient), 1) {
		o.Gradients[idx] = 1e99
	} else if !math.IsNaN(float64(newGradient)) {
		o.Gradients[idx] = newGradient
	}
}

// Adjust returns the update for a given weight and adjusts learnig rate based on gradint signs
func (o *SGD) Adjust(synapse *deep.Synapse, k, iteration, idx int, E, E_1 deep.Deepfloat64) (deep.Deepfloat64, bool, bool) {
	var newValue deep.Deepfloat64
	fakeRoot := false
	completed := false

	if iteration > 2 {
		value := synapse.Weights[k]
		value_1 := synapse.Weights_1[k]
		fx := o.Gradients[idx]
		fx_1 := o.Gradients_1[idx]
		d := (value - value_1) / (fx - fx_1)

		//if idx == 0 {
		//	fmt.Printf("idx: %v;; d:%v; fx: %v; fx_1: %v\n", idx, d, fx, fx_1)
		//	fmt.Printf("\tvalue: %v; value_1: %v\n", value, value_1)
		//	fmt.Printf("\tE: %v; E_1: %v\n", E, E_1)
		//}
		if math.Abs(float64(fx)) < deep.Eps && d > 0 && !math.IsInf(float64(d), 0) {
			completed = true
			newValue = 0
		} else if (iteration & 1) != 0 {
			//newValue = -d * fx
			newValue = -E / fx
		} else {
			tau := (E_1*E_1 + o.Lrs[idx]*E*E) / (E_1*E_1 + E*E)
			newValue = tau * o.Moments[idx]
			//if idx == 0 {
			//	fmt.Printf("\tMoments: %v; tau: %v; newValue: %v\n", o.Moments[idx], tau, newValue)
			//}
		}
		if math.IsNaN(float64(newValue)) || math.IsInf(float64(newValue), 0) {
			newValue = -o.Lrs[idx] * o.Gradients[idx]
			//} else {
			//	o.Lrs[idx] = d
		}
		//fakeRoot = math.Abs(float64(fx)) < deep.Eps && !math.Signbit(float64(-d))
		//if idx == 0 {
		//	fmt.Printf("\tfakeRoot: %v; newValue: %v\n", fakeRoot, newValue)
		//}
	} else {
		newValue = -o.Lrs[idx] * o.Gradients[idx]
	}
	if !math.IsNaN(float64(newValue)) {
		o.Moments[idx] = newValue
	}

	if math.Signbit(float64(o.Gradients[idx])) != math.Signbit(float64(o.Gradients_1[idx])) {
		o.Lrs[idx] *= 0.95
	} else {
		o.Lrs[idx] *= 1 / 0.95
	}
	o.Gradients_1[idx] = o.Gradients[idx]

	//if idx == 0 {
	//	fmt.Printf("\tAdjust: %v;\t   newWeight: %v\n", o.Moments[idx], synapse.Weights[k]+o.Moments[idx])
	//}
	return o.Moments[idx], fakeRoot, completed
}

// Gradient returns gradient value for an index
func (o *SGD) Gradient(idx int) float64 {
	return float64(o.Gradients[idx])
}

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
func (o *Adam) Init(size int) {
	o.v, o.m = make([]float64, size), make([]float64, size)
	o.gradints = make([]float64, size)
}

// Initialise gradients
func (o *Adam) InitGradients() {

}

// Adjust learning rates based on gradient signs
func (o *Adam) Adjust(synapse *deep.Synapse, k, t, idx int, E, E_1 deep.Deepfloat64) (deep.Deepfloat64, bool, bool) {
	//value := synapse.Weights[k]
	gradient := o.gradints[idx]

	lrt := deep.Deepfloat64(o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, float64(t)))) /
		(1.0 - math.Pow(o.beta, float64(t))))
	o.m[idx] = o.beta*o.m[idx] + (1.0-o.beta)*float64(gradient)
	o.v[idx] = o.beta2*o.v[idx] + (1.0-o.beta2)*math.Pow(float64(gradient), 2.0)

	return -lrt * deep.Deepfloat64(o.m[idx]/(math.Sqrt(o.v[idx])+o.epsilon)), false, false
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
