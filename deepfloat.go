package deep

import (
	"strconv"
)

type Deepfloat64 float64

func (f Deepfloat64) MarshalJSON() ([]byte, error) {
	return []byte(strconv.FormatFloat(float64(f), 'e', -1, 64)), nil
}
