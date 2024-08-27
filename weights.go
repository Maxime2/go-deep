package deep

import (
	"math/rand"
	"time"
)

type WeightType int

const (
	// WeightDefault is unspecified weight function type
	WeightDefault WeightType = 0
	// WeightUniform is uniform weight function type
	WeightUniform WeightType = 1
	// WeightNormal is normal weight function type
	WeightNormal WeightType = 2
	// WeightIdentity is to generate 0,1,0,1,0,1,...
	WeightIdentity WeightType = 3
)

// A WeightInitializer returns a (random) weight
type WeightInitializer func() Deepfloat64

func GetWeightFunction(wt WeightType, stdDev, mean float64) WeightInitializer {
	switch wt {
	case WeightUniform:
		return NewUniform(stdDev, mean)
	case WeightNormal:
		return NewNormal(stdDev, mean)
	case WeightIdentity:
		return NewIdentity()
	}
	return NewNormal(0.999, 0)
}

var wI Deepfloat64 = 0

// NewIdentity returns a identity weight generator
func NewIdentity() WeightInitializer {
	return func() Deepfloat64 { r := wI; wI = 1 - wI; return r }
}

// NewUniform returns a uniform weight generator
func NewUniform(stdDev, mean float64) WeightInitializer {
	rand.Seed(time.Now().UnixNano())
	return func() Deepfloat64 { return Deepfloat64(Uniform(stdDev, mean)) }
}

// Uniform samples a value from u(mean-stdDev/2,mean+stdDev/2)
func Uniform(stdDev, mean float64) float64 {
	return (rand.Float64()-0.5)*stdDev + mean

}

// NewNormal returns a normal weight generator
func NewNormal(stdDev, mean float64) WeightInitializer {
	rand.Seed(time.Now().UnixNano())
	return func() Deepfloat64 { return Deepfloat64(Normal(stdDev, mean)) }
}

// Normal samples a value from N(μ, σ)
func Normal(stdDev, mean float64) float64 {
	return rand.NormFloat64()*stdDev + mean
}
