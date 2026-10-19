# Reaction Yield Prediction with Neural Network Regression

Project 9 (Reliance Chemicals Ltd. case study). A PyTorch neural network predicts chemical reaction yield (%) from temperature, pressure and reactant concentration.

**Test-set results (300 unseen reactions):**

| Model | R² | MAE (%) |
|---|---|---|
| **Neural network (YieldNet)** | **0.991** | **1.80** |
| Random forest | 0.987 | 2.14 |
| Quadratic RSM-style regression | 0.717 | 11.52 |
| Linear regression | 0.192 | 19.21 |

## Run it

```bash
pip install -r requirements.txt
npm install                                   # only needed to rebuild the Word report
python run_experiments.py                     # data, training, baselines, ablations, CV, figures (~2 min)
python predict.py --temperature 392 --pressure 25 --concentration 1.0
node report/build_report.js                   # rebuild the Word report from the latest results
```

Results are reproducible: seed 42 is fixed, so every run produces identical numbers.

## Layout

| Path | What it is |
|---|---|
| `configs/config.json` | All hyperparameters |
| `src/data_generator.py` | Physically motivated reaction simulator (used because Reaxys is licensed) |
| `src/preprocess.py` | 70/15/15 split and strict feature scaling (scaler fitted on the training set only) |
| `src/model.py` | YieldNet: 3 → 64 → 32 → 1, ReLU, linear output |
| `src/train.py` | Adam + MSE + early stopping |
| `src/metrics.py`, `src/plots.py` | R², MAE, RMSE and all figures |
| `run_experiments.py` | Runs the full study |
| `predict.py` | Yield prediction for new conditions |
| `results/` | `metrics.json`, CSV tables, `figures/`, trained `model/` |
| `report/Reaction_Yield_Prediction_Report.docx` | The project report (structure follows the whiteboard page plan) |
| `viva_notes.md` | Answers to the three viva questions, plus likely follow-ups |
