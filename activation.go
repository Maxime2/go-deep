package deep

import (
	"math"
)

// Mode denotes inference mode
type Mode int

const (
	// ModeDefault is unspecified mode
	ModeDefault Mode = 0
	// ModeMultiClass is for one-hot encoded classification, applies softmax output layer
	ModeMultiClass Mode = 1
	// ModeRegression is regression, applies linear output layer
	ModeRegression Mode = 2
	// ModeBinary is binary classification, applies sigmoid output layer
	ModeBinary Mode = 3
	// ModeMultiLabel is for multilabel classification, applies sigmoid output layer
	ModeMultiLabel Mode = 4
)

// OutputActivation returns activation corresponding to prediction mode
func OutputActivation(c Mode) ActivationType {
	switch c {
	case ModeMultiClass:
		return ActivationSoftmax
	case ModeRegression:
		return ActivationLinear
	case ModeBinary, ModeMultiLabel:
		return ActivationSigmoid
	}
	return ActivationNone
}

// GetActivation returns the concrete activation given an ActivationType
func GetActivation(act ActivationType) Differentiable {
	switch act {
	case ActivationSigmoid:
		return Sigmoid{}
	case ActivationTanh:
		return Tanh{}
	case ActivationReLU:
		return ReLU{}
	case ActivationLinear:
		return Linear{}
	case ActivationSoftmax:
		return Linear{}
	}
	return Linear{}
}

// ActivationType is represents a neuron activation function
type ActivationType int

const (
	// ActivationNone is no activation
	ActivationNone ActivationType = 0
	// ActivationSigmoid is a sigmoid activation
	ActivationSigmoid ActivationType = 1
	// ActivationTanh is hyperbolic activation
	ActivationTanh ActivationType = 2
	// ActivationReLU is rectified linear unit activation
	ActivationReLU ActivationType = 3
	// ActivationLinear is linear activation
	ActivationLinear ActivationType = 4
	// ActivationSoftmax is a softmax activation (per layer)
	ActivationSoftmax ActivationType = 5
)

// Differentiable is an activation function and its first order derivative,
// where the latter is expressed as a function of the former for efficiency
type Differentiable interface {
	F(Deepfloat64) Deepfloat64
	Df(Deepfloat64) Deepfloat64
	If(Deepfloat64) Deepfloat64
	Idomain(y, ideal Deepfloat64) Deepfloat64
}

// Sigmoid is a logistic activator in the special case of a = 1
type Sigmoid struct{}

// F is Sigmoid(x)
func (a Sigmoid) F(x Deepfloat64) Deepfloat64 { return Logistic(x, 1) }

// Df is Sigmoid'(y), where y = Sigmoid(x)
func (a Sigmoid) Df(y Deepfloat64) Deepfloat64 { return y * (1 - y) }

// If is inverse to Sigmoid
func (a Sigmoid) If(y Deepfloat64) Deepfloat64 { return Deepfloat64(math.Log(float64(y / (1 - y)))) }

// Idomain() ensures the value is in the domain of activation inverse function, (0,1)
func (a Sigmoid) Idomain(y, ideal Deepfloat64) Deepfloat64 {
	if ideal < Eps {
		ideal = 0.2*Eps + 0.8*y
	} else if ideal > 1-Eps {
		ideal = 0.2*(1-Eps) + 0.8*y
	}
	return ideal
}

// Logistic is the logistic function
func Logistic(x, a Deepfloat64) Deepfloat64 {
	if a*x > 36 {
		return 0.9999999999999999
	}
	if a*x < -709 {
		return 1.216780750623423e-308
	}
	exponent := math.Exp(float64(-a * x))
	if math.IsInf(exponent, 1) {
		return 0.9999999999999999
	}
	if math.IsInf(exponent, -1) {
		return 0.0000000000000001
	}
	r := 1.0 / (1.0 + exponent)
	return Deepfloat64(r)
}

// Tanh is a hyperbolic activator
type Tanh struct{}

// F is Tanh(x)
func (a Tanh) F(x Deepfloat64) Deepfloat64 {
	return (1 - Deepfloat64(math.Exp(float64(-2*x)))) / (1 + Deepfloat64(math.Exp(float64(-2*x))))
}

// Df is Tanh'(y), where y = Tanh(x)
func (a Tanh) Df(y Deepfloat64) Deepfloat64 { return 1 - Deepfloat64(math.Pow(float64(y), 2)) }

// If is Artanh(y)
func (a Tanh) If(y Deepfloat64) Deepfloat64 {
	return Deepfloat64(0.5 * math.Log(float64(1+y)/float64(1-y)))
}

// Idomain() ensures the value is in the domain of activation inverse function
func (a Tanh) Idomain(y, ideal Deepfloat64) Deepfloat64 {
	if ideal < -1+Eps {
		return 0.1*(-1+Eps) + 0.9*y
	}
	if ideal > 1-Eps {
		return 0.1*(1-Eps) + 0.9*y
	}
	return ideal
}

// ReLU is a rectified linear unit activator
type ReLU struct{}

// F is ReLU(x)
func (a ReLU) F(x Deepfloat64) Deepfloat64 { return Deepfloat64(math.Max(float64(x), 0)) }

// Df is ReLU'(y), where y = ReLU(x)
func (a ReLU) Df(y Deepfloat64) Deepfloat64 {
	if y > 0 {
		return 1
	}
	return 0
}

// If is inverse to ReLU(), to some extent
func (a ReLU) If(y Deepfloat64) Deepfloat64 { return y }

// Idomain() ensures the value is in the domain of activation inverse function
func (a ReLU) Idomain(y, ideal Deepfloat64) Deepfloat64 { return ideal }

// Linear is a linear activator
type Linear struct{}

// F is the identity function
func (a Linear) F(x Deepfloat64) Deepfloat64 { return x }

// Df is constant
func (a Linear) Df(x Deepfloat64) Deepfloat64 { return 1 }

// If is reverse to identity
func (a Linear) If(y Deepfloat64) Deepfloat64 { return y }

// Idomain() ensures the value is in the domain of activation inverse function
func (a Linear) Idomain(y, ideal Deepfloat64) Deepfloat64 { return ideal }
