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
	w      *tabwriter.Writer
	prefix string
}

// NewStatsPrinter creates a StatsPrinter
func NewStatsPrinter(precision int) *StatsPrinter {
	width := precision + 10
	if width < 16 {
		width = 16
	}
	return &StatsPrinter{tabwriter.NewWriter(log.Writer(), width, 0, 2, ' ', 0), ""}
}

// SetPrefix set new prefix
func (p *StatsPrinter) SetPrefix(prefix string) {
	p.prefix = prefix
}

// Init initializes printer
func (p *StatsPrinter) Init(n *deep.Neural) {
	fmt.Fprintf(p.w, "%s\tEpochs\tElapsed\tError\tLoss (%s)\t", p.prefix, n.Config.Loss)
	if n.Config.Mode == deep.ModeMultiClass {
		fmt.Fprintf(p.w, "Accuracy\t\n%s\t---\t---\t---\t---\t---\t\n", p.prefix)
	} else {
		fmt.Fprintf(p.w, "\n%s\t---\t---\t---\t---\t\n", p.prefix)
	}
	p.w.Flush()
}

// PrintProgress prints the current state of training
func (p *StatsPrinter) PrintProgress(n *deep.Neural, validation Examples, elapsed time.Duration, iteration uint32, completed float64) {
	fmt.Fprintf(p.w, "%s\t%d (%d)\t%s\t%.*e\t%.*e\t%s\n", p.prefix,
		iteration, n.Config.Epoch+1, //completed,
		elapsed.String(),
		n.Config.LossPrecision, n.TotalError,
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

	return deep.GetLoss(n.Config.Loss).Cf(predictions, responses)
}
