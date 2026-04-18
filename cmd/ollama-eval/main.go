package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	ollamaeval "github.com/JonasEnglund/ollama-eval"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	if command == "help" || command == "-h" || command == "--help" {
		printUsage()
		os.Exit(0)
	}
	if command != "run" {
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", command)
		printUsage()
		os.Exit(1)
	}

	var (
		datasetPath string
		models      []string
		judgeModel  string
		outputPath  string
		method      string
		endpoint    string
	)

	i := 2
	for i < len(os.Args) {
		arg := os.Args[i]
		i++
		switch arg {
		case "-d", "--dataset":
			if i < len(os.Args) {
				datasetPath = os.Args[i]
				i++
			}
		case "-m", "--models":
			for i < len(os.Args) && !isFlag(os.Args[i]) {
				models = append(models, os.Args[i])
				i++
			}
		case "-j", "--judge":
			if i < len(os.Args) {
				judgeModel = os.Args[i]
				i++
			}
		case "-o", "--output":
			if i < len(os.Args) {
				outputPath = os.Args[i]
				i++
			}
		case "-t", "--method":
			if i < len(os.Args) {
				method = os.Args[i]
				i++
			}
		case "-e", "--endpoint":
			if i < len(os.Args) {
				endpoint = os.Args[i]
				i++
			}
		default:
			fmt.Fprintf(os.Stderr, "unknown flag: %s\n", arg)
			printUsage()
			os.Exit(1)
		}
	}

	if datasetPath == "" {
		fmt.Fprintln(os.Stderr, "error: --dataset required")
		os.Exit(1)
	}
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "error: --models required")
		os.Exit(1)
	}
	if outputPath == "" {
		outputPath = "results.html"
	}
	if method == "" {
		method = "similarity"
	}
	if endpoint == "" {
		endpoint = "http://localhost:11434"
	}

	dataset, err := ollamaeval.LoadDataset(datasetPath)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	client := ollamaeval.NewOllamaClient(endpoint)

	var scorer ollamaeval.Scorer
	switch method {
	case "llm":
		if judgeModel == "" {
			log.Fatalf("--judge required for method=llm")
		}
		scorer = ollamaeval.NewJudgeScorer(client, judgeModel)
	case "both":
		if judgeModel == "" {
			log.Fatalf("--judge required for method=both")
		}
		scorer = ollamaeval.NewBothScorer(client, judgeModel)
	default:
		scorer = ollamaeval.NewEmbeddingScorer()
	}

	runner := ollamaeval.NewRunner(endpoint, models, scorer)
	runner.OnProgress = progressBar

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fmt.Println("Evaluating...")
	results, err := runner.Run(ctx, dataset)
	fmt.Println()
	if err != nil {
		log.Fatalf("run evaluation: %v", err)
	}

	if err := ollamaeval.GenerateReport(results, models, method, outputPath); err != nil {
		log.Fatalf("generate report: %v", err)
	}

	fmt.Printf("results written to %s\n", outputPath)
}

func progressBar(done, total int) {
	const width = 30
	pct := float64(done) / float64(total)
	filled := int(pct * float64(width))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	fmt.Printf("\r[%s] %3d%% (%d/%d)", bar, int(pct*100), done, total)
}

func isFlag(s string) bool {
	return len(s) > 0 && s[0] == '-'
}

func printUsage() {
	fmt.Print(`Usage: ollama-eval run [flags]

Flags:
  -d, --dataset <path>   Path to dataset JSONL file (required)
  -m, --models  <names>  One or more model names to evaluate (required)
  -j, --judge   <model>  Judge model for llm/both scoring methods
  -o, --output  <path>   Output file path (default: results.html)
  -t, --method  <name>   Scoring method: similarity | llm | both (default: similarity)
  -e, --endpoint <url>   Ollama API endpoint (default: http://localhost:11434)
  -h, --help             Show this help

Examples:
  ollama-eval run -d examples/tickets.jsonl -m llama3:latest
  ollama-eval run -d examples/tickets.jsonl -m llama3:latest qwen3:14b
  ollama-eval run -d examples/tickets.jsonl -m llama3:latest -j llama3:latest -t both
`)
}
