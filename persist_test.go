package deep

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_RestoreFromDump(t *testing.T) {
	rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{5, 3, 1},
		Activation: ActivationSigmoid,
		Weight:     NewUniform(0.5, 0),
		Bias:       true,
	})

	dump := n.Dump()
	new := FromDump(dump)

	for i, biases := range n.Biases {
		for j, bias := range biases {
			for k := 0; k < len(bias.Weights); k++ {
				assert.Equal(t, bias.Weights[k], new.Biases[i][j].Weights[k])
			}
		}
	}
	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Predict([]Deepfloat64{0}), new.Predict([]Deepfloat64{0}))
}

func Test_Marshal(t *testing.T) {
	rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{3, 3, 1},
		Activation: ActivationSigmoid,
		Weight:     NewUniform(0.5, 0),
		Bias:       true,
	})

	dump, err := n.Marshal()
	assert.Nil(t, err)

	new, err := Unmarshal(dump)
	assert.Nil(t, err)

	for i, biases := range n.Biases {
		for j, bias := range biases {
			for k := 0; k < len(bias.Weights); k++ {
				assert.Equal(t, bias.Weights[k], new.Biases[i][j].Weights[k])
			}
		}
	}
	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Predict([]Deepfloat64{0}), new.Predict([]Deepfloat64{0}))
}
