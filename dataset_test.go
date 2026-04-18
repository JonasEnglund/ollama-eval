package ollamaeval

import (
	"os"
	"testing"
)

func TestLoadDataset(t *testing.T) {
	content := `{"input": "question one", "expected": "answer one"}
{"input": "question two", "expected": "answer two"}
`
	f, err := os.CreateTemp("", "dataset-*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
	f.Close()

	dataset, err := LoadDataset(f.Name())
	if err != nil {
		t.Fatalf("LoadDataset error: %v", err)
	}

	if len(dataset) != 2 {
		t.Fatalf("expected 2 data points, got %d", len(dataset))
	}
	if dataset[0].Input != "question one" {
		t.Errorf("dataset[0].Input = %q, want %q", dataset[0].Input, "question one")
	}
	if dataset[1].Expected != "answer two" {
		t.Errorf("dataset[1].Expected = %q, want %q", dataset[1].Expected, "answer two")
	}
}

func TestLoadDatasetNotFound(t *testing.T) {
	_, err := LoadDataset("/nonexistent/path.jsonl")
	if err == nil {
		t.Error("expected error for nonexistent file, got nil")
	}
}

func TestLoadDatasetInvalidJSON(t *testing.T) {
	f, err := os.CreateTemp("", "bad-*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	f.WriteString("not json\n")
	f.Close()

	_, err = LoadDataset(f.Name())
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
}
