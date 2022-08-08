package deep

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/kylelemons/godebug/pretty"
	"github.com/stretchr/testify/assert"
)

func Test_Init(t *testing.T) {
	n := NewNeural(&Config{
		Inputs:     3,
		Layout:     []int{4, 4, 2},
		Activation: ActivationTanh,
		Mode:       ModeBinary,
		Weight:     NewUniform(0.5, 0),
		Bias:       true,
	})

	assert.Len(t, n.Layers, len(n.Config.Layout))
	for i, l := range n.Layers {
		assert.Len(t, l.Neurons, n.Config.Layout[i])
	}
}

func Test_Forward_Save_Load(t *testing.T) {
	c := Config{
		Inputs:     3,
		Layout:     []int{3, 3, 3},
		Activation: ActivationReLU,
		Mode:       ModeMultiClass,
		Weight:     NewNormal(1.0, 0),
		Bias:       true,
	}
	n := NewNeural(&c)
	weights := [][][]deepfloat64{
		{
			{0.1, 0.4, 0.3},
			{0.3, 0.7, 0.7},
			{0.5, 0.2, 0.9},
		},
		{
			{0.2, 0.3, 0.5},
			{0.3, 0.5, 0.7},
			{0.6, 0.4, 0.8},
		},
		{
			{0.1, 0.4, 0.8},
			{0.3, 0.7, 0.2},
			{0.5, 0.2, 0.9},
		},
	}
	for _, n := range n.Layers[1].Neurons {
		n.A = ActivationSigmoid
	}
	for i, l := range n.Layers {
		for j, n := range l.Neurons {
			for k := 0; k < 3; k++ {
				n.In[k].Weight0 = weights[i][j][k]
				n.In[k].Weight1 = weights[i][j][k]
				n.In[k].Weight2 = weights[i][j][k]
			}
		}
	}
	for _, biases := range n.Biases {
		for _, bias := range biases {
			bias.Weight0 = 1
			bias.Weight1 = 1
			bias.Weight2 = 1
		}
	}

	err := n.Forward([]deepfloat64{0.1, 0.2, 0.7})
	assert.Nil(t, err)

	expected := [][]float64{
		{1.3, 1.66, 1.72},
		{0.9320110830223464, 0.9684462334302945, 0.9785427102823965},
		{0.31106226665743886, 0.27860738455524936, 0.4103303487873119},
	}
	for i := range n.Layers {
		for j, n := range n.Layers[i].Neurons {
			assert.InEpsilon(t, expected[i][j], float64(n.Value), 1e-12)
		}
	}

	tmpfile, err := ioutil.TempFile("", "test_load_save")
	assert.Nil(t, err)
	defer os.Remove(tmpfile.Name()) // clean up

	err = n.Save(tmpfile.Name())
	assert.Nil(t, err)

	n2 := NewNeural(&c)
	err = n2.Load(tmpfile.Name())
	assert.Nil(t, err)

	if diff := pretty.Compare(n, n2); diff != "" {
		t.Errorf("n and n2 diff: (-got +want)\n%s", diff)
	}

	assert.Equal(t, true, reflect.DeepEqual(n, n2), "the networks should be the same")

	err = n.Forward([]deepfloat64{0.1, 0.2})
	assert.Error(t, err)
}

func Test_NumWeights(t *testing.T) {
	n := NewNeural(&Config{Layout: []int{5, 5, 3}})
	assert.Equal(t, 5+5*6+3*5, n.NumWeights())
}
