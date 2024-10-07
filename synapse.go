package deep

import (
	"fmt"
	"math"

	tabulatedfunction "github.com/Maxime2/tabulated-function"
)

type SynapseType int

const (
	// SynapseAnalytic is an analytical function
	SynapseTypeAnalytic SynapseType = 0
	// SynapseTabulated is a tabulated function
	SynapseTypeTabulated SynapseType = 1
)

// Synapse is an edge between neurons
type Synapse interface {
	Refire()
	Fire(Deepfloat64)
	FireDerivative() Deepfloat64
	SetTag(string)
	GetTag() string
	SetWeight(int, Deepfloat64)
	GetGradient(Deepfloat64, int) Deepfloat64
	GetWeight(int) Deepfloat64
	FireDelta(Deepfloat64) Deepfloat64
	SetIsBias(bool)
	String() string
	WeightsString() string
	GetIn() Deepfloat64
	GetOut() Deepfloat64
	Len() int
	SetWeights([]Deepfloat64)
	GetWeights() []Deepfloat64
	GetUp() *Neuron
	Epoch(uint32)
	AddPoint(x, y Deepfloat64, it uint32)
	GetPoint(i int) (Deepfloat64, Deepfloat64)
	DrawPS(path string)
}

type SynapseTabulated struct {
	direct     *tabulatedfunction.TabulatedFunction
	inverse    *tabulatedfunction.TabulatedFunction
	derivative *tabulatedfunction.TabulatedFunction
	changed    bool
	Up         *Neuron
	In, Out    Deepfloat64
	IsBias     bool
	Tag        string
}

// NewSynapseTabulated returns a tabulated function synapse preset with specific tag
func NewSynapseTabulated(up *Neuron, tag string) *SynapseTabulated {
	direct := tabulatedfunction.New()
	direct.SetOrder(1)
	inverse := tabulatedfunction.New()
	inverse.SetOrder(1)
	derivative := tabulatedfunction.New()
	syn := &SynapseTabulated{
		direct:     direct,
		inverse:    inverse,
		derivative: derivative,
		In:         0,
		Out:        0,
		IsBias:     false,
		Tag:        tag,
		Up:         up,
	}
	syn.direct.LoadConstant(0.5, 0, 1)
	syn.inverse.LoadConstant(0.5, 0.5, 0.5)
	return syn
}

func (s *SynapseTabulated) String() string {
	return fmt.Sprintf("Tag: %v; In: %v; Out: %v; isBias: %v; %v",
		s.Tag, s.In, s.Out, s.IsBias, s.direct.String())
}

func (s *SynapseTabulated) WeightsString() string {
	return "[tabulated]"
}

func (s *SynapseTabulated) Refire() {
	s.Out = Deepfloat64(s.direct.F(float64(s.In)))
}

func (s *SynapseTabulated) Fire(value Deepfloat64) {
	s.In = value
	s.Refire()
}

func (s *SynapseTabulated) FireDerivative() Deepfloat64 {
	return Deepfloat64(s.derivative.F(float64(s.In)))
}

func (s *SynapseTabulated) SetTag(tag string) {
	s.Tag = tag
}

func (s *SynapseTabulated) GetTag() string {
	return s.Tag
}

func (s *SynapseTabulated) SetWeight(k int, weight Deepfloat64) {

}

func (s *SynapseTabulated) GetGradient(D_E_x Deepfloat64, k int) Deepfloat64 {
	return D_E_x * Deepfloat64(math.Pow(float64(s.In), float64(k)))
}

func (s *SynapseTabulated) GetWeight(k int) Deepfloat64 {
	return Deepfloat64(math.NaN())
}

func (s *SynapseTabulated) Epoch(epoch uint32) {
	s.direct.Epoch(epoch)
	s.inverse.Epoch(epoch)
	s.derivative.Epoch(epoch)
}

func (s *SynapseTabulated) FireDelta(D_E_x Deepfloat64) Deepfloat64 {
	//	mul := Deepfloat64(1)
	//	var res Deepfloat64
	//	for k := 0; k < len(s.Weights); k++ {
	//		res += mul * s.GetGradient(D_E_x, k)
	//		mul *= s.In
	//	}
	//	return res
	return Deepfloat64(math.NaN())
}

func (s *SynapseTabulated) SetIsBias(value bool) {
	s.IsBias = value
}

func (s *SynapseTabulated) GetIn() Deepfloat64 {
	return s.In
}

func (s *SynapseTabulated) GetOut() Deepfloat64 {
	return s.Out
}

func (s *SynapseTabulated) Len() int {
	return s.direct.GetNdots()
}

func (s *SynapseTabulated) SetWeights(w []Deepfloat64) {}
func (s *SynapseTabulated) GetWeights() []Deepfloat64 {
	return []Deepfloat64{}
}

func (s *SynapseTabulated) GetUp() *Neuron {
	return s.Up
}

func (s *SynapseTabulated) AddPoint(x, y Deepfloat64, it uint32) {
	y_inserted := s.direct.AddPoint(float64(x), float64(y), it)
	s.inverse.AddPoint(y_inserted, float64(x), it)
	s.changed = true
}

// GetPoint() returns n-th point in Tabulated activation
func (s *SynapseTabulated) GetPoint(i int) (Deepfloat64, Deepfloat64) {
	return Deepfloat64(s.direct.P[i].X), Deepfloat64(s.direct.P[i].Y)
}

func (s *SynapseTabulated) DrawPS(path string) {
	s.direct.DrawPS(path)
}

type SynapseAnalytic struct {
	Weights []Deepfloat64
	Up      *Neuron
	In, Out Deepfloat64
	IsBias  bool
	Tag     string
}

// NewSynapseAnalytic returns a synapse with the weigths preset with specified initializer
// and marked with specified tag
func NewSynapseAnalytic(up *Neuron, degree int, weight WeightInitializer, tag string) *SynapseAnalytic {
	var weights = make([]Deepfloat64, degree+1)
	for i := 0; i <= degree; i++ {
		weights[i] = weight()
	}
	return &SynapseAnalytic{
		Weights: weights,
		In:      0,
		Out:     0,
		IsBias:  false,
		Tag:     tag,
		Up:      up,
	}
}

func (s *SynapseAnalytic) String() string {
	return fmt.Sprintf("Tag: %v; In: %v; Out: %v; isBias: %v; Weights: %v",
		s.Tag, s.In, s.Out, s.IsBias, s.Weights)
}

func (s *SynapseAnalytic) WeightsString() string {
	return fmt.Sprintf("%v", s.Weights)
}

func (s *SynapseAnalytic) Refire() {
	mul := Deepfloat64(1)
	s.Out = 0
	for k := 0; k < len(s.Weights); k++ {
		s.Out += s.Weights[k] * mul
		mul *= s.In
	}
}

func (s *SynapseAnalytic) Fire(value Deepfloat64) {
	s.In = value
	s.Refire()
}

func (s *SynapseAnalytic) FireDerivative() Deepfloat64 {
	mul := Deepfloat64(1)
	var res Deepfloat64
	for k := 1; k < len(s.Weights); k++ {
		res += Deepfloat64(k) * mul * s.Weights[k]
		mul *= s.In
	}
	return res
}

func (s *SynapseAnalytic) SetTag(tag string) {
	s.Tag = tag
}
func (s *SynapseAnalytic) GetTag() string {
	return s.Tag
}

func (s *SynapseAnalytic) SetWeight(k int, weight Deepfloat64) {
	s.Weights[k] = weight
}

func (s *SynapseAnalytic) GetGradient(D_E_x Deepfloat64, k int) Deepfloat64 {
	return D_E_x * Deepfloat64(math.Pow(float64(s.In), float64(k)))
}

func (s *SynapseAnalytic) GetWeight(k int) Deepfloat64 {
	return s.Weights[k]
}

func (s *SynapseAnalytic) FireDelta(D_E_x Deepfloat64) Deepfloat64 {
	mul := Deepfloat64(1)
	var res Deepfloat64
	for k := 0; k < len(s.Weights); k++ {
		res += mul * s.GetGradient(D_E_x, k)
		mul *= s.In
	}
	return res
}
func (s *SynapseAnalytic) SetIsBias(value bool) {
	s.IsBias = value
}

func (s *SynapseAnalytic) GetIn() Deepfloat64 {
	return s.In
}

func (s *SynapseAnalytic) GetOut() Deepfloat64 {
	return s.Out
}

func (s *SynapseAnalytic) Len() int {
	return len(s.Weights)
}

func (s *SynapseAnalytic) SetWeights(w []Deepfloat64) {
	s.Weights = w
}

func (s *SynapseAnalytic) GetWeights() []Deepfloat64 {
	return s.Weights
}

func (s *SynapseAnalytic) GetUp() *Neuron {
	return s.Up
}

func (s *SynapseAnalytic) Epoch(uint32) {}

func (s *SynapseAnalytic) AddPoint(x, y Deepfloat64, it uint32) {}

// GetPoint() returns (0,0,1)
func (s *SynapseAnalytic) GetPoint(i int) (Deepfloat64, Deepfloat64) { return 0, 0 }

func (s *SynapseAnalytic) DrawPS(path string) {}
