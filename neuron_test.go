package deep

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Synapse_Fire(t *testing.T) {
	s := &SynapseAnalytic{
		Weights: []Deepfloat64{277.9848903746018,
			-46.494014333557395,
			-52.90141723171177,
			-53.09001803405063,
			-53.10138885017573,
			-53.10279737262996,
			-53.10308684399308,
			-53.10317433993435},
		In:     7.9556310898783e-14,
		Out:    0,
		IsBias: false,
		Tag:    "test",
	}
	s.Fire(7.9556310898783e-14)
	assert.InEpsilon(t, float64(s.Out), 277.984890374598, 0.0000001)
}
