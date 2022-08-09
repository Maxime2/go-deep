package deep

import "math"

// Mean of xx
func Mean(xx []Deepfloat64) Deepfloat64 {
	var sum Deepfloat64
	for _, x := range xx {
		sum += x
	}
	return sum / Deepfloat64(len(xx))
}

// Variance of xx
func Variance(xx []Deepfloat64) Deepfloat64 {
	if len(xx) == 1 {
		return 0.0
	}
	m := Mean(xx)

	var variance float64
	for _, x := range xx {
		variance += math.Pow(float64(x-m), 2)
	}

	return Deepfloat64(variance / float64(len(xx)-1))
}

// StandardDeviation of xx
func StandardDeviation(xx []Deepfloat64) Deepfloat64 {
	return Deepfloat64(math.Sqrt(float64(Variance(xx))))
}

// Standardize (z-score) shifts distribution to μ=0 σ=1
func Standardize(xx []Deepfloat64) {
	m := Mean(xx)
	s := StandardDeviation(xx)

	if s == 0 {
		s = 1
	}

	for i, x := range xx {
		xx[i] = (x - m) / s
	}
}

// Normalize scales to (0,1)
func Normalize(xx []Deepfloat64) {
	min, max := Min(xx), Max(xx)
	for i, x := range xx {
		xx[i] = (x - min) / (max - min)
	}
}

// Min is the smallest element
func Min(xx []Deepfloat64) Deepfloat64 {
	min := xx[0]
	for _, x := range xx {
		if x < min {
			min = x
		}
	}
	return min
}

// Max is the largest element
func Max(xx []Deepfloat64) Deepfloat64 {
	max := xx[0]
	for _, x := range xx {
		if x > max {
			max = x
		}
	}
	return max
}

// ArgMax is the index of the largest element
func ArgMax(xx []Deepfloat64) int {
	max, idx := xx[0], 0
	for i, x := range xx {
		if x > max {
			max, idx = xx[i], i
		}
	}
	return idx
}

// Sgn is signum
func Sgn(x float64) float64 {
	switch {
	case x < 0:
		return -1.0
	case x > 0:
		return 1.0
	}
	return 0
}

// Sum is sum
func Sum(xx []Deepfloat64) (sum Deepfloat64) {
	for _, x := range xx {
		sum += x
	}
	return
}

// Softmax is the softmax function
func Softmax(xx []Deepfloat64) []Deepfloat64 {
	out := make([]Deepfloat64, len(xx))
	var sum Deepfloat64
	max := Max(xx)
	for i, x := range xx {
		out[i] = Deepfloat64(math.Exp(float64(x - max)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Round to nearest integer
func Round(x Deepfloat64) Deepfloat64 {
	return Deepfloat64(math.Floor(float64(x) + .5))
}

// Dot product
func Dot(xx, yy []float64) float64 {
	var p float64
	for i := range xx {
		p += xx[i] * yy[i]
	}
	return p
}
