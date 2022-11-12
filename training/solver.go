package training

import (
	"math"

	deep "github.com/Maxime2/go-deep"
)

// Solver implements an update rule for training a NN
type Solver interface {
	Init(size int)
	Update(value, gradient, in deep.Deepfloat64, iteration, idx int) deep.Deepfloat64
}

// SGD is stochastic gradient descent with nesterov/momentum
type SGD struct {
	lr       float64
	decay    float64
	momentum float64
	//nesterov bool
	moments []deep.Deepfloat64
}

// NewSGD returns a new SGD solver
func NewSGD(lr, momentum, decay float64, nesterov bool) *SGD {
	return &SGD{
		lr:       fparam(lr, 0.01),
		decay:    decay,
		momentum: momentum,
		//nesterov: nesterov,
	}
}

// Init initializes vectors using number of weights in network
func (o *SGD) Init(size int) {
	o.moments = make([]deep.Deepfloat64, size)
}

// Update returns the update for a given weight
func (o *SGD) Update(value, gradient, in deep.Deepfloat64, iteration, idx int) deep.Deepfloat64 {
	lr := deep.Deepfloat64(o.lr / (1 + o.decay*float64(iteration)))
	lr = lr / (1 + lr*in*in)

	o.moments[idx] = deep.Deepfloat64(o.momentum)*o.moments[idx] - lr*gradient
	//
	//	if o.nesterov {
	//		o.moments[idx] = deep.Deepfloat64(o.momentum)*o.moments[idx] - lr*gradient
	//	}

	return o.moments[idx]
}

// Adam is an Adam solver
type Adam struct {
	lr      float64
	beta    float64
	beta2   float64
	epsilon float64

	v, m []float64
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
}

// Update returns the update for a given weight
func (o *Adam) Update(value, gradient, in deep.Deepfloat64, t, idx int) deep.Deepfloat64 {
	lrt := deep.Deepfloat64(o.lr * (math.Sqrt(1.0 - math.Pow(o.beta2, float64(t)))) /
		(1.0 - math.Pow(o.beta, float64(t))))
	o.m[idx] = o.beta*o.m[idx] + (1.0-o.beta)*float64(gradient)
	o.v[idx] = o.beta2*o.v[idx] + (1.0-o.beta2)*math.Pow(float64(gradient), 2.0)

	return -lrt * deep.Deepfloat64(o.m[idx]/(math.Sqrt(o.v[idx])+o.epsilon))
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
