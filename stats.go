package deep

import (
	"fmt"
	"math"
	"os"
	"sort"
)

type InputStats struct {
	Min, Avg, Max []Deepfloat64
	count         []int
	totalAvg      Deepfloat64
}

func NewInputStats(degree int) *InputStats {
	return &InputStats{
		Min:   make([]Deepfloat64, degree+1),
		Max:   make([]Deepfloat64, degree+1),
		Avg:   make([]Deepfloat64, degree+1),
		count: make([]int, degree+1),
	}
}

func (s *InputStats) Init(k int, in Deepfloat64) {
	in = Deepfloat64(math.Abs(float64(in)))
	s.Min[k] = in
	s.Max[k] = in
	s.Avg[k] = in
	s.count[k] = 1
}

func (s *InputStats) Update(k int, in Deepfloat64) {
	in = Deepfloat64(math.Abs(float64(in)))
	if in < s.Min[k] {
		s.Min[k] = in
	} else if in > s.Max[k] {
		s.Max[k] = in
	}
	s.Avg[k] += in
	s.count[k]++
}

func (s *InputStats) isNew(k int) bool {
	return s.count[k] == 0
}

func (s *InputStats) Finalise() {
	s.totalAvg = 0
	for i := 0; i < len(s.count); i++ {
		s.Avg[i] /= Deepfloat64(s.count[i])
		if i > 0 {
			s.totalAvg += s.Avg[i]
		}
	}
	s.totalAvg /= Deepfloat64(len(s.count) - 1)
}

func (n *Neural) InputStats(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	stats := map[string]*InputStats{}
	Layer := n.Layers[0]
	for _, neuron := range Layer.Neurons {
		for _, syn := range neuron.In {
			s, ok := stats[syn.Tag]
			if !ok {
				stats[syn.Tag] = NewInputStats(n.Config.Degree)
				s = stats[syn.Tag]
			}
			for k := 0; k <= n.Config.Degree; k++ {
				if s.isNew(k) {
					s.Init(k, syn.Weights[k])
				} else {
					s.Update(k, syn.Weights[k])
				}
			}
		}
	}
	for _, s := range stats {
		s.Finalise()
	}
	keys := make([]string, 0, len(stats))

	for key := range stats {
		keys = append(keys, key)
	}

	sort.SliceStable(keys, func(i, j int) bool {
		return stats[keys[i]].totalAvg < stats[keys[j]].totalAvg
	})

	for _, key := range keys {
		fmt.Fprintf(f, "%s : totalAvg: %v\n", key, stats[key].totalAvg)
		for k := 0; k <= n.Config.Degree; k++ {
			fmt.Fprintf(f, "\tk=%d : Avg: %v;  Min %v; Max: %v\n", k, stats[key].Avg[k], stats[key].Min[k], stats[key].Max[k])
		}
	}

	return nil
}
