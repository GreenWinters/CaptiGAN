import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENT_DIR = "experiments/experiment_20251112_210616"

def plot_loss_curves(history_path, out_dir):
    history = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history['D_loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['D_loss'], label='Discriminator Loss')
    plt.plot(epochs, history['G_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DCGAN Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcgan_loss_curves.png"))
    plt.close()
    print(f"Saved loss curves to {os.path.join(out_dir, 'dcgan_loss_curves.png')}")

def plot_fid_curve(history_path, out_dir):
    history = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history['FID']) + 1)
    fids = [f if f is not None else np.nan for f in history['FID']]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fids, marker='o', label='FID')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.title('FID Score Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcgan_fid_curve.png"))
    plt.close()
    print(f"Saved FID curve to {os.path.join(out_dir, 'dcgan_fid_curve.png')}")

def summarize_optuna_results(optuna_results_path, out_dir):
    with open(optuna_results_path, 'rb') as f:
        results = pickle.load(f)
    # Only include trials with 'best_fid'
    rows = []
    for trial in results:
        if 'best_fid' in trial and 'params' in trial:
            row = trial['params'].copy()
            row['trial_number'] = trial.get('trial_number', None)
            row['best_fid'] = trial['best_fid']
            rows.append(row)
        else:
            print(f"Warning: Skipping trial without 'best_fid' or 'params': {trial}")
    if not rows:
        print("No valid trials with 'best_fid' found in optuna_results.pkl.")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values('best_fid')
    df.to_csv(os.path.join(out_dir, "optuna_trials_summary.csv"), index=False)
    print(f"Saved Optuna trial summary table to {os.path.join(out_dir, 'optuna_trials_summary.csv')}")
    print(df.head(10).to_markdown(index=False, tablefmt='github'))

def summarize_optuna_study(optuna_study_path, out_dir):
    with open(optuna_study_path, 'rb') as f:
        study = pickle.load(f)
    best_trial = study.best_trial
    print("\nBest Optuna Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Value (Best FID): {best_trial.value}")
    print(f"  Params: {json.dumps(best_trial.params, indent=2)}")
    # Save best trial info
    with open(os.path.join(out_dir, "best_optuna_trial.json"), "w") as f:
        json.dump({
            "trial_number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params
        }, f, indent=2)
    print(f"Saved best trial info to {os.path.join(out_dir, 'best_optuna_trial.json')}")

def print_all_optuna_trials(optuna_study_path):
    with open(optuna_study_path, 'rb') as f:
        study = pickle.load(f)
    print("\nAll Optuna Trials Summary:")
    print(f"{'Trial':<8} {'Value (FID)':<15} {'Params'}")
    for trial in study.trials:
        print(f"{trial.number:<8} {trial.value:<15.6f} {trial.params}")

if __name__ == "__main__":
    history_path = os.path.join(EXPERIMENT_DIR, "dcgan_loss_history.npy")
    optuna_results_path = os.path.join(EXPERIMENT_DIR, "optuna_results.pkl")
    optuna_study_path = os.path.join(EXPERIMENT_DIR, "optuna_study.pkl")
    out_dir = EXPERIMENT_DIR

    if os.path.exists(history_path):
        plot_loss_curves(history_path, out_dir)
        plot_fid_curve(history_path, out_dir)
    else:
        print(f"Could not find {history_path}")

    if os.path.exists(optuna_results_path):
        summarize_optuna_results(optuna_results_path, out_dir)
    else:
        print(f"Could not find {optuna_results_path}")

    if os.path.exists(optuna_study_path):
        summarize_optuna_study(optuna_study_path, out_dir)
        print_all_optuna_trials(optuna_study_path)
    else:
        print(f"Could not find {optuna_study_path}")