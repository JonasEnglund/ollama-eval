// Package ollamaeval provides tools for evaluating and comparing Ollama LLM models
// against a labeled dataset. It supports multiple scoring strategies and generates
// HTML or JSON reports.
//
// # Quick start
//
//	dataset, _ := ollamaeval.LoadDataset("examples/tickets.jsonl")
//	client := ollamaeval.NewOllamaClient("http://localhost:11434")
//	scorer := ollamaeval.NewEmbeddingScorer()
//	runner := ollamaeval.NewRunner("http://localhost:11434", []string{"llama3:latest"}, scorer)
//	results, _ := runner.Run(context.Background(), dataset)
//	ollamaeval.GenerateReport(results, []string{"llama3:latest"}, "similarity", "report.html")
package ollamaeval

// EvalConfig holds the configuration for a single evaluation run.
type EvalConfig struct {
	Models      []string
	JudgeModel  string
	DatasetPath string
	OutputPath  string
	Method      string
	Endpoint    string
}
