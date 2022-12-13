package training

import (
	"fmt"
	"log"
	"text/tabwriter"
	"time"

	deep "github.com/Maxime2/go-deep"
)

// StatsPrinter prints training progress
type StatsPrinter struct {
	w *tabwriter.Writer
}

// NewStatsPrinter creates a StatsPrinter
func NewStatsPrinter() *StatsPrinter {
	return &StatsPrinter{tabwriter.NewWriter(log.Writer(), 16, 0, 3, ' ', 0)}
}

// Init initializes printer
func (p *StatsPrinter) Init(n *deep.Neural) {
	fmt.Fprintf(p.w, "Epochs\tElapsed\tLoss (%s)\t", n.Config.Loss)
	if n.Config.Mode == deep.ModeMultiClass {
		fmt.Fprintf(p.w, "Accuracy\t\n---\t---\t---\t---\t\n")
	} else {
		fmt.Fprintf(p.w, "\n---\t---\t---\t\n")
	}
}

// PrintProgress prints the current state of training
func (p *StatsPrinter) PrintProgress(n *deep.Neural, validation Examples, elapsed time.Duration, iteration int) {
	fmt.Fprintf(p.w, "%d (%d)\t%s\t%.*e\t%s\n",
		iteration, n.Config.N_iterations,
		elapsed.String(),
		n.Config.LossPrecision, crossValidate(n, validation),
		formatAccuracy(n, validation))
	p.w.Flush()
}

func formatAccuracy(n *deep.Neural, validation Examples) string {
	if n.Config.Mode == deep.ModeMultiClass {
		return fmt.Sprintf("%.2f\t", accuracy(n, validation))
	}
	return ""
}

func accuracy(n *deep.Neural, validation Examples) float64 {
	correct := 0
	for _, e := range validation {
		est := n.Predict(e.Input)
		if deep.ArgMax(e.Response) == deep.ArgMax(est) {
			correct++
		}
	}
	return float64(correct) / float64(len(validation))
}

func crossValidate(n *deep.Neural, validation Examples) deep.Deepfloat64 {
	predictions, responses := make([][]deep.Deepfloat64, len(validation)), make([][]deep.Deepfloat64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}

	return deep.GetLoss(n.Config.Loss).F(predictions, responses)
}
