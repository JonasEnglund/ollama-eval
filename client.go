package ollamaeval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type generateRequest struct {
	Model  string          `json:"model"`
	Prompt string          `json:"prompt"`
	Stream bool            `json:"stream"`
	Format json.RawMessage `json:"format,omitempty"`
}

type generateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// OllamaClient is a minimal HTTP client for the Ollama API.
type OllamaClient struct {
	baseURL string
	client  *http.Client
}

// NewOllamaClient creates a client pointing at the given base URL (e.g. "http://localhost:11434").
func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 10 * time.Minute},
	}
}

// Generate sends a prompt to model and returns the response text.
func (c *OllamaClient) Generate(ctx context.Context, model, prompt string) (string, error) {
	return c.generate(ctx, model, prompt, nil)
}

// GenerateWithFormat sends a prompt with a JSON schema format constraint.
func (c *OllamaClient) GenerateWithFormat(ctx context.Context, model, prompt, format string) (string, error) {
	return c.generate(ctx, model, prompt, json.RawMessage(format))
}

func (c *OllamaClient) generate(ctx context.Context, model, prompt string, format json.RawMessage) (string, error) {
	req := generateRequest{Model: model, Prompt: prompt, Stream: false, Format: format}
	payload, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/generate", bytes.NewBuffer(payload))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	var result generateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return result.Response, nil
}
