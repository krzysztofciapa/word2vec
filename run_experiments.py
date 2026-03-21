import os
import subprocess
import time
import sys
import argparse
import shutil

def run_experiments():
    parser = argparse.ArgumentParser(description="Run comparison experiments on multiple processors")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--optimizers", type=str, nargs="+", default=['sgd', 'adagrad', 'sgld'],
                        help="List of optimizers to run (separated by space)")
    parser.add_argument("--burn-in-epochs", type=int, default=0, help="SGLD burn-in epochs (if non-zero, overrides --burn-in-frac)")
    args = parser.parse_args()
    
    optimizers = args.optimizers
    epochs = args.epochs
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # clean only logs/results for active optimizers
    for opt in optimizers:
        for folder in ['logs', 'results']:
            prefix = "train_" if folder == "logs" else "eval_"
            suffix = ".log" if folder == "logs" else ".json"
            fpath = os.path.join(folder, f"{prefix}{opt}{suffix}")
            if os.path.exists(fpath):
                os.remove(fpath)
    
    processes = []
    
    print(f"Starting parallel training for optimizers: {', '.join(optimizers)}")
    print(f"Epochs: {epochs}")
    print(f"This will launch {len(optimizers)} parallel processes. Output will be saved to logs/ directory.")
    
    start_time = time.time()
    
    for opt in optimizers:
        model_path = f"models/model_{opt}.npz"
        log_path = f"logs/train_{opt}.log"
        
        cmd = [
            sys.executable, "-u", "src/train.py",
            "--optimizer", opt,
            "--epochs", str(epochs),
            "--output", model_path
        ]
        
        if opt == 'sgld' and args.burn_in_epochs > 0:
            cmd.extend(["--burn-in-epochs", str(args.burn_in_epochs)])
        
        log_file = open(log_path, "w", buffering=1)
        log_file.truncate(0)  # explicitly clear 
        print(f"Launching {opt} -> saving to {model_path} (logs: {log_path})")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((opt, p, log_file, model_path))

    print("\nAll processes launched. Training in progress...\n")
    

    for _ in optimizers:
        print()
        
    active_processes = list(processes)
    status_lines = {opt: f"[{opt.upper()}] Starting..." for opt, _, _, _ in processes}
    
    failed = False
    
    while active_processes:
        
        sys.stdout.write("\033[F" * len(optimizers))
        
        for opt in optimizers:
            log_path = f"logs/train_{opt}.log"
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            for line in reversed(lines):
                                line = line.strip()
                                if "Loss" in line or "summary" in line:
                                    status_lines[opt] = f"[{opt.upper():<7}] {line[:80]}"
                                    break
                except Exception:
                    pass
            
            sys.stdout.write(f"\033[K{status_lines[opt]}\n")
            
        sys.stdout.flush()
        
        # check if processes are still running
        still_active = []
        for opt, p, log_file, model_path in active_processes:
            if p.poll() is None:
                still_active.append((opt, p, log_file, model_path))
            else:
                log_file.close()
                if p.returncode != 0:
                    status_lines[opt] = f"[{opt.upper():<7}] FAILED with exit code {p.returncode}."
                    failed = True
                else:
                    status_lines[opt] = f"[{opt.upper():<7}] Training finished successfully."
        active_processes = still_active
        time.sleep(1.0)
        
    sys.stdout.write("\033[F" * len(optimizers))
    for opt in optimizers:
        sys.stdout.write(f"\033[K{status_lines[opt]}\n")
    sys.stdout.flush()
            
    total_time = time.time() - start_time
    print(f"\nAll training runs completed in {total_time/60:.2f} minutes.")
    
    if failed:
        print("Some runs failed. Skipping evaluation.")
        return
        
    print("\n" + "="*50)
    print("RUNNING EVALUATION ON ALL MODELS")
    print("="*50)
    
    for opt, _, _, model_path in processes:
        print(f"\nEvaluating {opt.upper()} model:")
        print("-" * 30)
        
        json_path = f"results/eval_{opt}.json"
        
        cmd = [
            sys.executable, "run_eval.py",
            "--weights", model_path,
            "--analogies", 
            "--wordsim",
            "--save-json", json_path
        ]
        
        subprocess.run(cmd)

    print("\n" + "*"*50)
    print("EVALUATIONS COMPLETE.")
    print("Run `python analyze_results.py` to generate comparative charts from results/")
    print("*"*50)

if __name__ == "__main__":
    run_experiments()
