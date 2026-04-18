package ollamaeval

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     string
		wantMin  float64
		wantMax  float64
	}{
		{
			name:    "identical strings",
			a:       "hello world",
			b:       "hello world",
			wantMin: 1.0,
			wantMax: 1.0,
		},
		{
			name:    "completely different strings",
			a:       "hello world",
			b:       "foo bar baz",
			wantMin: 0.0,
			wantMax: 0.0,
		},
		{
			name:    "partial overlap",
			a:       "reset your password",
			b:       "how to reset password",
			wantMin: 0.3,
			wantMax: 0.9,
		},
		{
			name:    "empty strings",
			a:       "",
			b:       "",
			wantMin: 0.0,
			wantMax: 0.0,
		},
		{
			name:    "one empty string",
			a:       "hello",
			b:       "",
			wantMin: 0.0,
			wantMax: 0.0,
		},
		{
			name:    "case insensitive",
			a:       "Hello World",
			b:       "hello world",
			wantMin: 1.0,
			wantMax: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			const eps = 1e-9
		if got < tt.wantMin-eps || got > tt.wantMax+eps {
				t.Errorf("cosineSimilarity(%q, %q) = %.4f, want [%.2f, %.2f]",
					tt.a, tt.b, got, tt.wantMin, tt.wantMax)
			}
			if math.IsNaN(got) {
				t.Errorf("cosineSimilarity returned NaN")
			}
		})
	}
}

func TestWordFreq(t *testing.T) {
	freq := wordFreq("hello hello world")
	if freq["hello"] != 2 {
		t.Errorf("expected hello=2, got %d", freq["hello"])
	}
	if freq["world"] != 1 {
		t.Errorf("expected world=1, got %d", freq["world"])
	}
}

func TestCleanJSON(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{`{"score": 0.8}`, `{"score": 0.8}`},
		{"```json\n{\"score\": 0.8}\n```", `{"score": 0.8}`},
		{"```\n{\"score\": 0.8}\n```", `{"score": 0.8}`},
		{"  {\"score\": 0.8}  ", `{"score": 0.8}`},
	}

	for _, tt := range tests {
		got := cleanJSON(tt.input)
		if got != tt.want {
			t.Errorf("cleanJSON(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
