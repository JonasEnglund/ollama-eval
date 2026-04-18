package ollamaeval

import (
	"encoding/json"
	"html/template"
	"os"
)

// Result holds the evaluation outcome for one DataPoint across all models.
type Result struct {
	Input     string          `json:"input"`
	Expected  string          `json:"expected"`
	Responses []ModelResponse `json:"responses"`
}

// ModelResponse is a single model's answer and its score.
type ModelResponse struct {
	Model       string  `json:"model"`
	Response    string  `json:"response"`
	Score       float64 `json:"score"`
	Explanation string  `json:"explanation,omitempty"`
}

// ModelSummary aggregates scores for one model across all data points.
type ModelSummary struct {
	Model        string  `json:"model"`
	AverageScore float64 `json:"average_score"`
	BestScore    float64 `json:"best_score"`
	WorstScore   float64 `json:"worst_score"`
	TotalItems   int     `json:"total_items"`
}

// Report is the full evaluation output.
type Report struct {
	Title            string         `json:"title"`
	ModelSummaries   []ModelSummary `json:"model_summaries"`
	Results          []Result       `json:"results"`
	EvaluationMethod string         `json:"evaluation_method"`
}

// GenerateReport writes an HTML or JSON report to outputPath based on its extension.
func GenerateReport(results []Result, models []string, method, outputPath string) error {
	modelScores := make(map[string][]float64)
	for _, r := range results {
		for _, resp := range r.Responses {
			modelScores[resp.Model] = append(modelScores[resp.Model], resp.Score)
		}
	}

	var summaries []ModelSummary
	for _, model := range models {
		scores := modelScores[model]
		if len(scores) == 0 {
			continue
		}
		var sum float64
		best, worst := scores[0], scores[0]
		for _, s := range scores {
			sum += s
			if s > best {
				best = s
			}
			if s < worst {
				worst = s
			}
		}
		summaries = append(summaries, ModelSummary{
			Model:        model,
			AverageScore: sum / float64(len(scores)),
			BestScore:    best,
			WorstScore:   worst,
			TotalItems:   len(scores),
		})
	}

	report := Report{
		Title:            "Ollama Evaluation Report",
		ModelSummaries:   summaries,
		Results:          results,
		EvaluationMethod: method,
	}

	if getExtension(outputPath) == "json" {
		return writeJSONReport(report, outputPath)
	}
	
	return writeHTMLReport(report, outputPath)
}

func getExtension(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '.' {
			return path[i+1:]
		}
	}

	return ""
}

func writeJSONReport(report Report, outputPath string) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(outputPath, data, 0644)
}

func writeHTMLReport(report Report, outputPath string) error {
	tmpl, err := template.New("report").Parse(htmlTemplate)
	if err != nil {
		return err
	}
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()
	return tmpl.Execute(file, report)
}

var htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{.Title}}</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6;padding:20px;background:#f5f5f5}
.container{max-width:1200px;margin:0 auto}
h1{color:#333;margin-bottom:20px}
.summary{background:white;padding:20px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
.summary h2{margin-bottom:15px}
table{width:100%;border-collapse:collapse}
th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}
th{background:#f8f8f8;font-weight:600}
.score{font-weight:bold}
.score.high{color:#22c55e}
.score.medium{color:#eab308}
.score.low{color:#ef4444}
.model-col{min-width:150px}
.input-section,.expected-section,.response-section{background:#f9f9f9;padding:10px;border-radius:4px;margin:5px 0;white-space:pre-wrap}
.result-card{background:white;padding:20px;border-radius:8px;margin-bottom:15px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
.model-badge{display:inline-block;background:#3b82f6;color:white;padding:4px 8px;border-radius:4px;font-size:12px;margin-right:10px}
</style>
</head>
<body>
<div class="container">
<h1>{{.Title}}</h1>
<div class="summary">
<h2>Summary</h2>
<table>
<tr><th>Model</th><th>Average</th><th>Best</th><th>Worst</th><th>Items</th></tr>
{{range .ModelSummaries}}
<tr><td class="model-col">{{.Model}}</td>
<td class="score {{if gt .AverageScore 0.8}}high{{else if gt .AverageScore 0.5}}medium{{else}}low{{end}}">{{printf "%.2f" .AverageScore}}</td>
<td>{{printf "%.2f" .BestScore}}</td>
<td>{{printf "%.2f" .WorstScore}}</td>
<td>{{.TotalItems}}</td>
</tr>
{{end}}
</table>
<p><strong>Method:</strong> {{.EvaluationMethod}}</p>
</div>
{{range .Results}}
<div class="result-card">
<h3>Input</h3>
<div class="input-section">{{.Input}}</div>
<h3>Expected</h3>
<div class="expected-section">{{.Expected}}</div>
<h3>Responses</h3>
{{range .Responses}}
<div class="model-badge">{{.Model}}</div>
<div class="response-section">
<strong>Response:</strong> {{.Response}}<br>
<strong>Score:</strong> <span class="score {{if gt .Score 0.8}}high{{else if gt .Score 0.5}}medium{{else}}low{{end}}">{{printf "%.2f" .Score}}</span>
{{if .Explanation}}<br><strong>Explanation:</strong> {{.Explanation}}{{end}}
</div>
{{end}}
</div>
{{end}}
</div>
</body>
</html>`
