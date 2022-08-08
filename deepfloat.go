package deep

import (
	"strconv"
)

type deepfloat64 float64

func (f deepfloat64) MarshalJSON() ([]byte, error) {
	return []byte(strconv.FormatFloat(float64(f), 'e', -1, 64)), nil
}
