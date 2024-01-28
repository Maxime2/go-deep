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
		Activation: []ActivationType{ActivationSigmoid},
		Weight:     WeightUniform,
	})

	dump := n.Dump()
	new := FromDump(dump)

	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Predict([]Deepfloat64{0}), new.Predict([]Deepfloat64{0}))
}

func Test_Marshal(t *testing.T) {
	rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{3, 3, 1},
		Activation: []ActivationType{ActivationSigmoid},
		Weight:     WeightUniform,
	})

	dump, err := n.Marshal()
	assert.Nil(t, err)

	new, err := Unmarshal(dump)
	assert.Nil(t, err)

	assert.Equal(t, n.String(), new.String())
	assert.Equal(t, n.Predict([]Deepfloat64{0}), new.Predict([]Deepfloat64{0}))
}
