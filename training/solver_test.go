package training

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/kylelemons/godebug/pretty"
	"github.com/stretchr/testify/assert"
)

func Test_Save_Load(t *testing.T) {
	s := NewSGD(0.1)
	s2 := NewSGD(0.2)

	s.Init(10)
	s2.Init(10)

	tmpfile, err := ioutil.TempFile("", "test_load_save")
	assert.Nil(t, err)
	defer os.Remove(tmpfile.Name()) // clean up

	err = s.Save(tmpfile.Name())
	assert.Nil(t, err)

	err = s2.Load(tmpfile.Name())
	assert.Nil(t, err)

	if diff := pretty.Compare(s, s2); diff != "" {
		t.Errorf("s and s2 diff: (-got +want)\n%s", diff)
	}
}
