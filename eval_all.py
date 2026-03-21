import os
import subprocess
import json

MODELS_DIR = "models"
RESULTS_DIR = "results"
EVAL_SCRIPT = "run_eval.py"

def run_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".npz")]
    print(f"Found {len(models)} models to evaluate: {models}")
    
    for model_file in models:
        model_name = model_file.replace("model_", "").replace(".npz", "")
        weights_path = os.path.join(MODELS_DIR, model_file)
        results_path = os.path.join(RESULTS_DIR, f"eval_{model_name}.json")
        
        print(f"\nEvaluating model: {model_name}...")
        cmd = [
            "python", EVAL_SCRIPT,
            "--weights", weights_path,
            "--wordsim",
            "--analogies",
            "--save-json", results_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully evaluated {model_name}. Results saved to {results_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model_name}: {e}")

if __name__ == "__main__":
    run_evaluation()
