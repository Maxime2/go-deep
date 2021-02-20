package deep

import (
	"encoding/json"
)

// Dump is a neural network dump
type Dump struct {
	Config  *Config
	Weights [][][][]float64
}

// ApplyWeights sets the weights from a three-dimensional slice
func (n *Neural) ApplyWeights(weights [][][][]float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				wLen := len(n.Layers[i].Neurons[j].In[k].Weight)
				for m := range n.Layers[i].Neurons[j].In[k].Weight {
					n.Layers[i].Neurons[j].In[k].Weight[m] = weights[i][j][k][m]
					n.Layers[i].Neurons[j].In[k].In[m] = weights[i][j][k][wLen+m]
				}
			}
		}
	}
}

// Weights returns all weights in sequence
func (n Neural) Weights() [][][][]float64 {
	weights := make([][][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			weights[i][j] = make([][]float64, len(n.In))
			for k, in := range n.In {
				wLen := len(in.Weight)
				weights[i][j][k] = make([]float64, 2*wLen)
				for m := range in.Weight {
					weights[i][j][k][m] = in.Weight[m]
					weights[i][j][k][wLen+m] = in.In[m]
				}
			}
		}
	}
	return weights
}

// Dump generates a network dump
func (n Neural) Dump() *Dump {
	return &Dump{
		Config:  n.Config,
		Weights: n.Weights(),
	}
}

// FromDump restores a Neural from a dump
func FromDump(dump *Dump) *Neural {
	n := NewNeural(dump.Config)
	n.ApplyWeights(dump.Weights)

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
