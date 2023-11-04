package deep

import (
	"strconv"
)

type Deepfloat64 float64

func (f Deepfloat64) MarshalJSON() ([]byte, error) {
	return []byte(`"` + strconv.FormatFloat(float64(f), 'e', -1, 64) + `"`), nil
}

func (fs *Deepfloat64) UnmarshalJSON(b []byte) error {
    if b[0] == '"' {
        b = b[1 : len(b)-1]
    }
    f, err := strconv.ParseFloat(string(b), 64)
    *fs = Deepfloat64(f)
    return err
}
