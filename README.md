# ollama-eval

A Go library and CLI tool for evaluating and comparing [Ollama](https://ollama.com) LLM models against a labeled dataset.

## Features

- Run multiple models against a dataset in parallel
- Three scoring methods: cosine similarity, LLM-as-judge, or both combined
- HTML and JSON report output

## Installation

### CLI

```bash
go install github.com/JonasEnglund/ollama-eval/cmd/ollama-eval@latest
```

### Library

```bash
go get github.com/JonasEnglund/ollama-eval
```

## CLI Usage

```
ollama-eval run [flags]

Flags:
  -d, --dataset <path>   Path to dataset JSONL file (required)
  -m, --models  <names>  One or more model names to evaluate (required)
  -j, --judge   <model>  Judge model for llm/both scoring methods
  -o, --output  <path>   Output file path (default: results.html)
  -t, --method  <name>   Scoring method: similarity | llm | both (default: similarity)
  -e, --endpoint <url>   Ollama API endpoint (default: http://localhost:11434)
```

**Examples:**

```bash
# Compare two models using cosine similarity
ollama-eval run -d examples/tickets.jsonl -m llama3:latest qwen3:14b

# Use an LLM judge
ollama-eval run -d examples/tickets.jsonl -m llama3:latest -j llama3:latest -t llm

# Both methods combined, output JSON
ollama-eval run -d examples/tickets.jsonl -m llama3:latest -j llama3:latest -t both -o results.json
```

## Dataset Format

A JSONL file where each line is a JSON object with `input` and `expected` fields:

```jsonl
{"input": "How do I reset my password?", "expected": "Go to settings and click reset password."}
{"input": "What is your refund policy?", "expected": "Full refunds within 30 days."}
```

## Library Usage

```go
import ollamaeval "github.com/JonasEnglund/ollama-eval"

dataset, err := ollamaeval.LoadDataset("examples/tickets.jsonl")

client := ollamaeval.NewOllamaClient("http://localhost:11434")
scorer := ollamaeval.NewEmbeddingScorer()

// or: ollamaeval.NewJudgeScorer(client, "llama3:latest")
// or: ollamaeval.NewBothScorer(client, "llama3:latest")

runner := ollamaeval.NewRunner("http://localhost:11434", []string{"llama3:latest"}, scorer)
results, err := runner.Run(context.Background(), dataset)

err = ollamaeval.GenerateReport(results, []string{"llama3:latest"}, "similarity", "report.html")
```

### Custom scorer

Implement the `Scorer` interface to plug in your own evaluation logic:

```go
type Scorer interface {
    Score(ctx context.Context, input, actual, expected string) (score float64, explanation string)
}
```

## Scoring Methods

| Method       | Description |
|-------------|-------------|
| `similarity` | Bag-of-words cosine similarity between actual and expected responses |
| `llm`        | Uses a judge model to score 0–1 with a brief explanation |
| `both`       | Average of similarity and LLM judge scores |

## Requirements

- Go 1.22+
- Ollama running locally (or accessible via `--endpoint`)
