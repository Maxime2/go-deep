package deep

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/kylelemons/godebug/pretty"
	"github.com/stretchr/testify/assert"
	"github.com/theothertomelliott/acyclic"
)

func Test_Init(t *testing.T) {
	n := NewNeural(&Config{
		Inputs:     3,
		Layout:     []int{4, 4, 2},
		Activation: []ActivationType{ActivationTanh},
		Mode:       ModeBinary,
		Weight:     WeightUniform,
	})

	assert.Len(t, n.Layers, len(n.Config.Layout))
	for i, l := range n.Layers {
		assert.Len(t, l.Neurons, n.Config.Layout[i])
	}
}

func Test_Forward(t *testing.T) {
	c := Config{
		Degree:     1,
		Inputs:     3,
		Layout:     []int{3, 3, 3},
		Activation: []ActivationType{ActivationReLU},
		Mode:       ModeMultiClass,
		Weight:     WeightNormal,
	}
	n := NewNeural(&c)
	weights := [][][]Deepfloat64{
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
		n.A = GetActivation(ActivationSigmoid)
	}
	for i, l := range n.Layers {
		for j, n := range l.Neurons {
			for k := 0; k < 3; k++ {
				n.In[k].SetWeight(0, 0)
				n.In[k].SetWeight(1, weights[i][j][k])
			}
		}
	}

	err := n.Forward([]Deepfloat64{0.1, 0.2, 0.7})
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

	err = n.Forward([]Deepfloat64{0.1, 0.2})
	assert.Error(t, err)
}

func Test_Save_Load(t *testing.T) {
	c := Config{
		Degree:     1,
		Inputs:     3,
		Layout:     []int{3, 3, 3},
		Activation: []ActivationType{ActivationReLU},
		Mode:       ModeMultiClass,
	}
	n := NewNeural(&c)
	weights := [][][]Deepfloat64{
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
	for i, l := range n.Layers {
		for j, n := range l.Neurons {
			for k := 0; k < 3; k++ {
				n.In[k].SetWeight(0, 0)
				n.In[k].SetWeight(1, weights[i][j][k])
			}
		}
	}

	tmpfile, err := ioutil.TempFile("", "test_load_save")
	assert.Nil(t, err)
	defer os.Remove(tmpfile.Name()) // clean up

	t.Log("Doing SaveReadable")
	err = n.SaveReadable(tmpfile.Name())
	assert.Nil(t, err)

	t.Log("Doing Save")
	err = n.Save(tmpfile.Name())
	assert.Nil(t, err)

	t.Log("Doing Load")
	n2, err := Load(tmpfile.Name())
	assert.Nil(t, err)

	err = acyclic.Check(n)
	if err != nil {
		t.Errorf("n has a cycle")
	}
	err = acyclic.Check(n2)
	if err != nil {
		t.Errorf("n2 has a cycle")
	}

	t.Log("Doing Compare")
	if diff := pretty.Compare(n, n2); diff != "" {
		t.Errorf("n and n2 diff: (-got +want)\n%s", diff)
	}
	t.Log("Doing test.dot")
	n.Dot("test.dot")
}

func Test_NumWeights(t *testing.T) {
	n := NewNeural(&Config{Layout: []int{5, 5, 3}, Degree: 1})
	assert.Equal(t, 2*(5*5+3*5), n.NumWeights())
}
