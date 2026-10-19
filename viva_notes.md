# Viva Defence Notes — Project 9: Reaction Yield Prediction

Short answers to the three questions on the project slide, each backed by a result from this project. Numbers come from `results/metrics.json`.

---

## Q1. "This is regression, not classification. How does your final output layer differ?"

**Answer in one line:** My last layer is **one neuron with no activation function** (linear/identity), trained with **mean squared error**. A classifier would end in **softmax (or sigmoid)** neurons and be trained with **cross-entropy**.

| | Classification | My model (regression) |
|---|---|---|
| Output neurons | One per class | **One** |
| Activation | Softmax / sigmoid | **None (identity)** |
| Output means | Probability of each class | **Yield in %** (any real number) |
| Loss | Cross-entropy | **MSE** |
| Metrics | Accuracy, F1 | **R², MAE, RMSE, residual plot** |

**Why no activation?** Softmax would squash the output into probabilities that sum to 1. Sigmoid would squash it into 0–1. Yield is a continuous quantity, so the network must be free to output any value.

**Likely follow-up: "Then can it predict a negative yield?"** Yes, slightly. The lowest test prediction was **−0.58 %**, which is within the 2 % measurement noise. `predict.py` clips the output to 0–100 %. An alternative is a sigmoid × 100 output, but that slows learning near 0 % and 100 %.

**Code to point at:** `src/model.py`, the line `layers.append(nn.Linear(width, 1))`.

---

## Q2. "Why is data scaling crucial when mixing temperature and concentration data?"

**Answer in one line:** Temperature is about **376 K** and concentration about **1 mol/L**. Without scaling, temperature dominates the gradients and the network fails to learn. **My ablation proves it: R² falls from 0.991 to 0.153 without scaling.**

The reasoning, step by step:
1. The gradient for a first-layer weight is proportional to its input value, so temperature weights get gradients about **360× larger** than concentration weights.
2. One learning rate can't suit both: it is too big for temperature weights and too small for concentration weights.
3. Large raw inputs also push ReLU units into extreme activations, which makes training unstable.
4. StandardScaler gives every input mean 0 and standard deviation 1, so each variable gets equal footing.

**"Strict" scaling:** the scaler is fitted on the **training set only**, then reused for validation, test and new predictions. Fitting it on all the data would leak test-set information into training (data leakage) and inflate the scores.

**Evidence to show:**
- Figure 3 shows the boxplots before and after scaling.
- Figure 10 shows the validation loss with and without scaling: it stalls near 1.0 without scaling and drops about 100× lower with it.
- Table 10 is the ablation table.

**Likely follow-up: "Did the random forest need scaling?"** No. Tree splits compare one feature against a threshold, so the feature's scale doesn't matter. Only distance- and gradient-based models need it.

---

## Q3. "Explain your residual plot."

**Answer in one line:** Residual = actual − predicted. In my plot the residuals form a **flat, random band around zero with constant width**, which means the model has no bias and has captured all the learnable structure. What's left is measurement noise.

Three things to check, with my numbers:

| Check | What a good plot shows | My result |
|---|---|---|
| Centred on zero? | No systematic over- or under-prediction | Mean residual = **0.21 %** ≈ 0 |
| Constant spread? (homoscedasticity) | Same scatter at low and high yield | corr(\|residual\|, predicted) = **0.06**; std ≈ 2.3 in every band |
| Any pattern? | No curve or funnel | None; **96 %** of points are inside ±2σ (≈ 95 % expected) |

- **Histogram:** it matches the true noise distribution N(0, 2²). The residual std is **2.33** against a true noise level of **2.0**, so the residuals are almost pure noise.
- **Contrast (Figure 8):** the quadratic RSM model's residuals form a **U-shape**, which is the classic sign of underfitting. It even predicts negative yields, as low as −29.7 %.
- **Honest weakness:** for predictions above 60 %, the mean residual is +0.78. The model slightly under-predicts near the yield peak, where the curve is sharpest. More training data near the optimum would fix this.

**How to read bad residual plots, in case they ask:**
- A curve (U-shape) means the model is too simple (underfitting).
- A funnel (widening spread) means the error grows with yield (heteroscedasticity); a log transform or a weighted loss helps.
- An offset from zero means the model is biased.

---

## Other questions to be ready for

- **Why simulated data?** Reaxys is a licensed Elsevier database, and the slide allows simulated data. The simulator is physically motivated (Arrhenius kinetics, pressure saturation, selectivity loss above 420 K) and fully documented, so the results are reproducible.
- **Why not just use the random forest (R² 0.987)?** The network is more accurate on every metric (MAE 1.80 vs 2.14). Its prediction surface is also smooth and differentiable, so it can be used with optimisers to find the best conditions. The random forest's predictions are step-like (see the response curves in Figure 9).
- **How do you know it isn't overfitting?**
  - The validation loss tracks the training loss.
  - Early stopping restored the best epoch (279).
  - 5-fold cross-validation gives R² = 0.9906 ± 0.0005.
  - The test set was used only once.
- **What is the best possible score?** The noise ceiling: the true noise-free function scored on the noisy labels gets RMSE 2.06. My model gets 2.34, only 14 % above that floor.
- **Which input matters most?** Temperature: removing it drops R² to about 0. Concentration has almost zero linear correlation (r = −0.02), yet removing it lowers R² to 0.965, so the network uses non-linear effects that a correlation analysis misses.
- **Why the 64–32 architecture?**
  - A single 32-unit layer reached R² 0.988.
  - 128–64–32 reached 0.9916 with 4.6× the parameters.
  - 64–32 (0.9912) balances accuracy and size.
- **Optimum found?** The network puts the best temperature at 392.0 K (78.0 %); the true optimum is 392.7 K (78.2 %). The quadratic RSM model says 388.2 K and only 58.3 %.
