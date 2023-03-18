package deep

import (
	"math/rand"
	"time"
)

// A WeightInitializer returns a (random) weight
type WeightInitializer func() Deepfloat64

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
