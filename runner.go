package ollamaeval

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// ModelRunner runs a dataset against multiple Ollama models concurrently.
type ModelRunner struct {
	client     *OllamaClient
	models     []string
	scorer     Scorer
	OnProgress func(done, total int) // called after each data point is evaluated
}

// NewRunner creates a ModelRunner for the given endpoint, models, and scorer.
func NewRunner(endpoint string, models []string, scorer Scorer) *ModelRunner {
	return &ModelRunner{
		client: NewOllamaClient(endpoint),
		models: models,
		scorer: scorer,
	}
}

// Run evaluates every DataPoint against all models and returns the results.
// Models are queried in parallel for each data point.
func (r *ModelRunner) Run(ctx context.Context, dataset Dataset) ([]Result, error) {
	results := make([]Result, 0, len(dataset))

	for _, dp := range dataset {
		result := Result{
			Input:    dp.Input,
			Expected: dp.Expected,
		}

		var wg sync.WaitGroup
		var mu sync.Mutex

		for _, model := range r.models {
			wg.Add(1)
			go func(model string) {
				defer wg.Done()

				resp, err := r.client.Generate(ctx, model, dp.Input)
				if err != nil {
					log.Printf("model %s: %v", model, err)
					return
				}

				score, explanation := r.scorer.Score(ctx, dp.Input, resp, dp.Expected)

				mu.Lock()
				result.Responses = append(result.Responses, ModelResponse{
					Model:       model,
					Response:    resp,
					Score:       score,
					Explanation: explanation,
				})
				mu.Unlock()
			}(model)
		}

		wg.Wait()

		if len(result.Responses) == 0 {
			return nil, fmt.Errorf("no responses for input: %q", dp.Input)
		}

		results = append(results, result)

		if r.OnProgress != nil {
			r.OnProgress(len(results), len(dataset))
		}
	}

	return results, nil
}
