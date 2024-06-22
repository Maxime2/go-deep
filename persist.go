package deep

import (
	"encoding/json"
)

// Point is a point in Tabulated activation
type Point struct {
	X, Y Deepfloat64
}

// Dump is a neural network dump
type Dump struct {
	Config      *Config
	Weights     [][][][]Deepfloat64
	Activations [][]Point
}

// ApplyWeights sets the weights from a four-dimensional slice
func (n *Neural) ApplyWeights(weights [][][][]Deepfloat64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				n.Layers[i].Neurons[j].In[k].Weights = weights[i][j][k]
			}
		}
	}
}

// Weights returns all weights in sequence
func (n Neural) Weights() [][][][]Deepfloat64 {
	weights := make([][][][]Deepfloat64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][][]Deepfloat64, len(l.Neurons))
		for j, n := range l.Neurons {
			weights[i][j] = make([][]Deepfloat64, len(n.In))
			for k, in := range n.In {
				weights[i][j][k] = in.Weights
			}
		}
	}
	return weights
}

func (n Neural) ApplyActivations(points [][]Point) {
	current := 0
	for _, l := range n.Layers {
		if l.A == ActivationTabulated {
			for _, n := range l.Neurons {
				npoints := len(points[current])
				for i := 0; i < npoints; i++ {
					n.A.AddPoint(points[current][i].X, points[current][i].Y)
				}
				current++
			}
		}
	}

}

// Activations() returns points of all Tabulated activations
func (n Neural) Activations() [][]Point {
	var activations [][]Point
	for _, l := range n.Layers {
		if l.A == ActivationTabulated {
			for _, n := range l.Neurons {
				points := n.A.Points()
				acts := make([]Point, points)
				for i := 0; i < points; i++ {
					acts[i].X, acts[i].Y = n.A.GetPoint(i)
				}
				activations = append(activations, acts)
			}
		}
	}
	return activations
}

// Dump generates a network dump
func (n Neural) Dump() *Dump {
	return &Dump{
		Config:      n.Config,
		Weights:     n.Weights(),
		Activations: n.Activations(),
	}
}

// FromDump restores a Neural from a dump
func FromDump(dump *Dump) *Neural {
	n := NewNeural(dump.Config)
	n.ApplyWeights(dump.Weights)
	n.ApplyActivations(dump.Activations)

	return n
}

// Marshal marshals to JSON from network
func (n Neural) Marshal() ([]byte, error) {
	return json.Marshal(n.Dump())
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}
