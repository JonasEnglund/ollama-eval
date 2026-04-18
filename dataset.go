package ollamaeval

import (
	"encoding/json"
	"os"
)

// DataPoint is a single evaluation sample.
type DataPoint struct {
	Input    string `json:"input"`
	Expected string `json:"expected"`
}

// Dataset is a slice of DataPoints.
type Dataset []DataPoint

// LoadDataset reads a JSONL file where each line is a DataPoint.
func LoadDataset(path string) (Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var dataset Dataset
	decoder := json.NewDecoder(file)
	for decoder.More() {
		var dp DataPoint
		if err := decoder.Decode(&dp); err != nil {
			return nil, err
		}
		dataset = append(dataset, dp)
	}

	return dataset, nil
}
