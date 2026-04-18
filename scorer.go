package ollamaeval

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// Scorer evaluates how well an actual response matches the expected output.
type Scorer interface {
	Score(ctx context.Context, input, actual, expected string) (score float64, explanation string)
}

// EmbeddingScorer scores responses using bag-of-words cosine similarity.
type EmbeddingScorer struct{}

func NewEmbeddingScorer() *EmbeddingScorer { return &EmbeddingScorer{} }

func (s *EmbeddingScorer) Score(_ context.Context, _, actual, expected string) (float64, string) {
	score := cosineSimilarity(actual, expected)
	return score, fmt.Sprintf("cosine similarity: %.2f", score)
}

// JudgeScorer uses an LLM to judge response quality on a 0–1 scale.
type JudgeScorer struct {
	client     *OllamaClient
	judgeModel string
}

func NewJudgeScorer(client *OllamaClient, judgeModel string) *JudgeScorer {
	return &JudgeScorer{client: client, judgeModel: judgeModel}
}

func (s *JudgeScorer) Score(ctx context.Context, input, actual, expected string) (float64, string) {
	prompt := fmt.Sprintf(
		"You are an evaluator.\nInput: %s\nExpected: %s\nActual: %s\nScore 0-1 in JSON: {\"score\": 0.0-1.0, \"explanation\": \"brief\"}",
		input, expected, actual,
	)

	respText, err := s.client.GenerateWithFormat(ctx, s.judgeModel, prompt, `{"type":"object"}`)
	if err != nil {
		return 0.5, fmt.Sprintf("judge error: %v", err)
	}

	var result struct {
		Score       float64 `json:"score"`
		Explanation string  `json:"explanation"`
	}
	if err := json.Unmarshal([]byte(cleanJSON(respText)), &result); err != nil {
		return 0.5, respText
	}

	return result.Score, result.Explanation
}

// BothScorer combines cosine similarity and LLM judge scores equally.
type BothScorer struct {
	client     *OllamaClient
	judgeModel string
}

func NewBothScorer(client *OllamaClient, judgeModel string) *BothScorer {
	return &BothScorer{client: client, judgeModel: judgeModel}
}

func (s *BothScorer) Score(ctx context.Context, input, actual, expected string) (float64, string) {
	simScore := cosineSimilarity(actual, expected)
	llmScore, explanation := NewJudgeScorer(s.client, s.judgeModel).Score(ctx, input, actual, expected)
	combined := (simScore + llmScore) / 2

	return combined, fmt.Sprintf("llm: %.2f, similarity: %.2f, combined: %.2f — %s", llmScore, simScore, combined, explanation)
}

// cosineSimilarity computes bag-of-words cosine similarity between two strings.
func cosineSimilarity(a, b string) float64 {
	aVec := wordFreq(a)
	bVec := wordFreq(b)

	var dot float64
	for w, c := range aVec {
		dot += float64(c * bVec[w])
	}

	var aMag, bMag float64
	for _, c := range aVec {
		aMag += float64(c * c)
	}
	for _, c := range bVec {
		bMag += float64(c * c)
	}

	aMag = math.Sqrt(aMag)
	bMag = math.Sqrt(bMag)
	if aMag == 0 || bMag == 0 {
		return 0
	}

	return dot / (aMag * bMag)
}

func wordFreq(s string) map[string]int {
	freq := make(map[string]int)
	for _, w := range strings.Fields(strings.ToLower(s)) {
		freq[w]++
	}

	return freq
}

func cleanJSON(resp string) string {
	resp = strings.TrimSpace(resp)
	resp = strings.TrimPrefix(resp, "```json")
	resp = strings.TrimPrefix(resp, "```")
	resp = strings.TrimSuffix(resp, "```")

	return strings.TrimSpace(resp)
}
