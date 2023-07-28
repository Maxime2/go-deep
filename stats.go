package deep

import (
	"fmt"
	"math"
	"os"
	"sort"
)

type InputStats struct {
	Avg          []Deepfloat64
	AvgMi, AvgPl []Deepfloat64
	count        []int
	countPl      []int
	totalAvg     Deepfloat64
	totalAvgMi   Deepfloat64
	totalAvgPl   Deepfloat64
}

func NewInputStats(degree int) *InputStats {
	return &InputStats{
		Avg:     make([]Deepfloat64, degree+1),
		AvgMi:   make([]Deepfloat64, degree+1),
		AvgPl:   make([]Deepfloat64, degree+1),
		count:   make([]int, degree+1),
		countPl: make([]int, degree+1),
	}
}

func (s *InputStats) Init(k int, in Deepfloat64) {
	if in < 0 {
		s.AvgMi[k] = in
		s.AvgPl[k] = 0
		s.countPl[k] = 0
	} else {
		s.AvgPl[k] = in
		s.AvgMi[k] = 0
		s.countPl[k] = 1
	}
	in = Deepfloat64(math.Abs(float64(in)))
	s.Avg[k] = in
	s.count[k] = 1
}

func (s *InputStats) Update(k int, in Deepfloat64) {
	if in < 0 {
		s.AvgMi[k] += in
	} else {
		s.AvgPl[k] += in
		s.countPl[k]++
	}
	in = Deepfloat64(math.Abs(float64(in)))
	s.Avg[k] += in
	s.count[k]++
}

func (s *InputStats) isNew(k int) bool {
	return s.count[k] == 0
}

func (s *InputStats) Finalise() {
	s.totalAvg = 0
	s.totalAvgPl, s.totalAvgMi = 0, 0
	for i := 0; i < len(s.count); i++ {
		s.Avg[i] /= Deepfloat64(s.count[i])
		if s.countPl[i] > 0 {
			s.AvgPl[i] /= Deepfloat64(s.countPl[i])
		}
		if s.countPl[i] != s.count[i] {
			s.AvgMi[i] /= Deepfloat64(s.count[i] - s.countPl[i])
		}
		if i > 0 {
			s.totalAvg += s.Avg[i]
			s.totalAvgMi += s.AvgMi[i]
			s.totalAvgPl += s.AvgPl[i]
		}
	}
	s.totalAvg /= Deepfloat64(len(s.count) - 1)
	s.totalAvgMi /= Deepfloat64(len(s.count) - 1)
	s.totalAvgPl /= Deepfloat64(len(s.count) - 1)
}

func (n *Neural) InputStats(detail bool, path string) error {
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

	for i, key := range keys {
		fmt.Fprintf(f, "%s : Avg: %v; Mi: %v; Pl: %v -- %d\n", key,
			stats[key].totalAvg, stats[key].totalAvgMi, stats[key].totalAvgPl, i)
		if detail {
			for k := 0; k <= n.Config.Degree; k++ {
				fmt.Fprintf(f, "\tk=%d : Avg: %v;  Mi %v; Pl: %v\n", k, stats[key].Avg[k], stats[key].AvgMi[k], stats[key].AvgPl[k])
			}
		}
	}

	return nil
}
