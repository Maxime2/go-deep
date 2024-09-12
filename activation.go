package deep

import (
	"math"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
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
func GetActivation(act ActivationType) Activation {
	switch act {
	case ActivationSigmoid:
		return &Sigmoid{}
	case ActivationTanh:
		return &Tanh{}
	case ActivationReLU:
		return &ReLU{}
	case ActivationLinear:
		return &Linear{}
	case ActivationSoftmax:
		return &Linear{}
	case ActivationTabulated:
		return newTabulated()
	}
	return &Linear{}
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
	// ActivationTabulated is a tabled function activation
	ActivationTabulated ActivationType = 6
)

// Activation is an activation function and its first order derivative,
// where the latter is expressed as a function of the former for efficiency
type Activation interface {
	F(Deepfloat64) Deepfloat64
	Df(Deepfloat64) Deepfloat64
	If(Deepfloat64) Deepfloat64
	Idomain(y, ideal Deepfloat64) Deepfloat64
	AddPoint(x, y Deepfloat64, it uint32, cnt uint64)
	Len() int
	GetPoint(i int) (Deepfloat64, Deepfloat64, uint64)
	Domain() (Deepfloat64, Deepfloat64)
	Epoch(uint32)
	String() string
}

// Tabulated is a tabulated activator
type Tabulated struct {
	direct, inverse *tabulatedfunction.TabulatedFunction
	derivative      *tabulatedfunction.TabulatedFunction
	changed         bool
}

func newTabulated() *Tabulated {
	direct := tabulatedfunction.New()
	direct.SetOrder(1)
	inverse := tabulatedfunction.New()
	inverse.SetOrder(1)
	derivative := tabulatedfunction.New()
	return &Tabulated{
		direct:     direct,
		inverse:    inverse,
		derivative: derivative,
		changed:    false,
	}
}

// Tabulated activation function
func (a *Tabulated) F(x Deepfloat64) Deepfloat64 {
	return Deepfloat64(a.direct.F(float64(x)))
}

// Inverse of tabulated activation function
func (a *Tabulated) If(x Deepfloat64) Deepfloat64 {
	return Deepfloat64(a.inverse.F(float64(x)))
}

// Derivative of tabulated activation function
func (a *Tabulated) Df(x Deepfloat64) Deepfloat64 {
	if a.changed {
		a.derivative.Assign(a.direct)
		a.derivative.Derivative()
		a.changed = false
	}
	return Deepfloat64(a.derivative.F(float64(x)))
}

// Idomain() ensures the value is in the domain of inverse tabulated activation function
func (a *Tabulated) Idomain(y, ideal Deepfloat64) Deepfloat64 { return ideal }

// AddPoint() adds new point into tabulated activation function
func (a *Tabulated) AddPoint(x, y Deepfloat64, it uint32, cnt uint64) {
	y_inserted := a.direct.AddPoint(float64(x), float64(y), it, cnt)
	a.inverse.AddPoint(y_inserted, float64(x), it, cnt)
	a.changed = true
}

// Len() returns the number of poionts in Tabulated activation
func (a *Tabulated) Len() int {
	return len(a.direct.P)
}

// GetPoint() returns n-th point in Tabulated activation
func (a *Tabulated) GetPoint(i int) (Deepfloat64, Deepfloat64, uint64) {
	return Deepfloat64(a.direct.P[i].X), Deepfloat64(a.direct.P[i].Y), a.direct.P[i].Cnt
}

// Domain() return pair of (minimum, maximum) values defining ("meaningful") domain
func (a *Tabulated) Domain() (Deepfloat64, Deepfloat64) {
	return 0, 10
}

// String() return textual representation of direct function
func (a *Tabulated) String() string {
	return a.direct.String()
}

// Epoch() set epoch for tabulated functions
func (a *Tabulated) Epoch(epoch uint32) {
	a.direct.Epoch(epoch)
	a.inverse.Epoch(epoch)
	a.derivative.Epoch(epoch)
}

// Sigmoid is a logistic activator in the special case of a = 1
type Sigmoid struct{}

// F is Sigmoid(x)
func (a *Sigmoid) F(x Deepfloat64) Deepfloat64 { return Logistic(x, 1) }

// Df is Sigmoid'(y), where y = Sigmoid(x)
func (a *Sigmoid) Df(y Deepfloat64) Deepfloat64 { return y * (1 - y) }

// If is inverse to Sigmoid
func (a *Sigmoid) If(y Deepfloat64) Deepfloat64 { return Deepfloat64(math.Log(float64(y / (1 - y)))) }

// Idomain() ensures the value is in the domain of activation inverse function, (0,1)
func (a *Sigmoid) Idomain(y, ideal Deepfloat64) Deepfloat64 {
	if ideal < Eps {
		ideal = 0.5*Eps + 0.5*y
	} else if ideal > 1-Eps {
		ideal = 0.5*(1-Eps) + 0.5*y
	}
	return ideal
}

// AddPoint() do nothing.
func (a *Sigmoid) AddPoint(x, y Deepfloat64, it uint32, cnt uint64) {}

// Len() returns 0
func (a *Sigmoid) Len() int { return 0 }

// GetPoint() returns (0,0,1)
func (a *Sigmoid) GetPoint(i int) (Deepfloat64, Deepfloat64, uint64) { return 0, 0, 1 }

// Domain() return pair of (minimum, maximum) values defining ("meaningful") domain
func (a *Sigmoid) Domain() (Deepfloat64, Deepfloat64) {
	return 0, 2
}

// String() rreturn "Sigmoid"
func (a *Sigmoid) String() string {
	return "Sigmoid"
}

// Epoch() set epoch for tabulated functions
func (a *Sigmoid) Epoch(epoch uint32) {
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
func (a *Tanh) F(x Deepfloat64) Deepfloat64 {
	return (1 - Deepfloat64(math.Exp(float64(-2*x)))) / (1 + Deepfloat64(math.Exp(float64(-2*x))))
}

// Df is Tanh'(y), where y = Tanh(x)
func (a *Tanh) Df(y Deepfloat64) Deepfloat64 { return 1 - Deepfloat64(math.Pow(float64(y), 2)) }

// If is Artanh(y)
func (a *Tanh) If(y Deepfloat64) Deepfloat64 {
	return Deepfloat64(0.5 * math.Log(float64(1+y)/float64(1-y)))
}

// Idomain() ensures the value is in the domain of activation inverse function
func (a *Tanh) Idomain(y, ideal Deepfloat64) Deepfloat64 {
	if ideal < -1+Eps {
		return 0.1*(-1+Eps) + 0.9*y
	}
	if ideal > 1-Eps {
		return 0.1*(1-Eps) + 0.9*y
	}
	return ideal
}

// AddPoint() do nothing.
func (a *Tanh) AddPoint(x, y Deepfloat64, it uint32, cnt uint64) {}

// Len() returns 0
func (a *Tanh) Len() int { return 0 }

// GetPoint() returns (0,0,1)
func (a *Tanh) GetPoint(i int) (Deepfloat64, Deepfloat64, uint64) { return 0, 0, 1 }

// Domain() return pair of (minimum, maximum) values defining ("meaningful") domain
func (a *Tanh) Domain() (Deepfloat64, Deepfloat64) {
	return 0, 1
}

// Epoch() set epoch for tabulated functions
func (a *Tanh) Epoch(epoch uint32) {
}

// String() return "Tanh"
func (a *Tanh) String() string {
	return "Tanh"
}

// ReLU is a rectified linear unit activator
type ReLU struct{}

// F is ReLU(x)
func (a *ReLU) F(x Deepfloat64) Deepfloat64 { return Deepfloat64(math.Max(float64(x), 0)) }

// Df is ReLU'(y), where y = ReLU(x)
func (a *ReLU) Df(y Deepfloat64) Deepfloat64 {
	if y > 0 {
		return 1
	}
	return 0
}

// If is inverse to ReLU(), to some extent
func (a *ReLU) If(y Deepfloat64) Deepfloat64 { return y }

// Idomain() ensures the value is in the domain of activation inverse function
func (a *ReLU) Idomain(y, ideal Deepfloat64) Deepfloat64 { return ideal }

// AddPoint() do nothing.
func (a *ReLU) AddPoint(x, y Deepfloat64, it uint32, cnt uint64) {}

// Len() returns 0
func (a *ReLU) Len() int { return 0 }

// GetPoint() returns (0,0,1)
func (a *ReLU) GetPoint(i int) (Deepfloat64, Deepfloat64, uint64) { return 0, 0, 1 }

// Domain() return pair of (minimum, maximum) values defining ("meaningful") domain
func (a *ReLU) Domain() (Deepfloat64, Deepfloat64) {
	return 0, 1
}

// Epoch() set epoch for tabulated functions
func (a *ReLU) Epoch(epoch uint32) {
}

// String() return "ReLU"
func (a *ReLU) String() string {
	return "ReLU"
}

// Linear is a linear activator
type Linear struct{}

// F is the identity function
func (a *Linear) F(x Deepfloat64) Deepfloat64 { return x }

// Df is constant
func (a *Linear) Df(x Deepfloat64) Deepfloat64 { return 1 }

// If is reverse to identity
func (a *Linear) If(y Deepfloat64) Deepfloat64 { return y }

// Idomain() ensures the value is in the domain of activation inverse function
func (a *Linear) Idomain(y, ideal Deepfloat64) Deepfloat64 { return ideal }

// AddPoint() do nothing.
func (a *Linear) AddPoint(x, y Deepfloat64, it uint32, cnt uint64) {}

// Len() returns 0
func (a *Linear) Len() int { return 0 }

// GetPoint() returns (0,0,1)
func (a *Linear) GetPoint(i int) (Deepfloat64, Deepfloat64, uint64) { return 0, 0, 1 }

// Domain() return pair of (minimum, maximum) values defining ("meaningful") domain
func (a *Linear) Domain() (Deepfloat64, Deepfloat64) {
	return 0, 1
}

// Epoch() set epoch for tabulated functions
func (a *Linear) Epoch(epoch uint32) {
}

// String() return "Linear"
func (a *Linear) String() string {
	return "Linear"
}
