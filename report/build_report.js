// Builds report/Reaction_Yield_Prediction_Report.docx from results/metrics.json and results/figures.
// Layout follows the whiteboard: p1 title, p2 abstract, p3-6 introduction, p7-9 methodology,
// p10-18 experiments & results, p19-20 conclusion / future work / references.
//
//   node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer,
  AlignmentType, HeadingLevel, LevelFormat, PageNumber, WidthType, ShadingType, BorderStyle,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "results", "figures");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "metrics.json"), "utf8"));
const OUT = path.join(__dirname, "Reaction_Yield_Prediction_Report.docx");

// ---------- numbers ----------
const f = (x, d = 2) => Number(x).toFixed(d);
const nn = M.neural_network;
const cmp = M.comparison;
const LIN = cmp["Linear Regression"], POLY = cmp["Polynomial Regression (deg 2, RSM-style)"], RF = cmp["Random Forest"];
const cv = M.cross_validation;
const ab = M.ablation;
const res = M.residuals;
const opt = M.optimum_temperature;
const ceil = M.noise_ceiling;
const sc = M.scaler;
const ds = M.dataset;
const sz = M.split_sizes;
const cfg = M.config;
const rmseGapPct = ((nn.rmse / ceil.rmse - 1) * 100).toFixed(0);
const rfMaeGapPct = ((RF.mae / nn.mae - 1) * 100).toFixed(0);

// ---------- layout constants (A4, 1-inch margins) ----------
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440, TEXT_W = PAGE_W - 2 * MARGIN; // 9026 DXA
const FONT = "Times New Roman";
const GREY = "555555";

// ---------- helpers ----------
// Inline markup: **bold**, *italic*, _{subscript}, ^{superscript} (bold/italic may contain sub/superscripts).
function rich(s, base = {}) {
  return s.split(/(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/).filter(Boolean).flatMap((p) => {
    if (p.startsWith("**")) return rich(p.slice(2, -2), { ...base, bold: true });
    if (p.startsWith("*")) return rich(p.slice(1, -1), { ...base, italics: true });
    if (p.startsWith("_{")) return [new TextRun({ ...base, text: p.slice(2, -1), subScript: true })];
    if (p.startsWith("^{")) return [new TextRun({ ...base, text: p.slice(2, -1), superScript: true })];
    return [new TextRun({ ...base, text: p })];
  });
}

// Figure and table numbers come from these orders, so text can reference items that appear later.
const FIG_ORDER = ["flowchart", "dist", "scaling", "yvf", "comparison", "trainfit", "residuals", "contrast", "response", "ablation"];
const TAB_ORDER = ["lit", "vars", "scaler", "output", "config", "preview", "test", "cv", "bands", "ablation", "future", "structure"];
const F = (k) => { const i = FIG_ORDER.indexOf(k); if (i < 0) throw new Error(`unknown figure ${k}`); return `Figure ${i + 1}`; };
const T = (k) => { const i = TAB_ORDER.indexOf(k); if (i < 0) throw new Error(`unknown table ${k}`); return `Table ${i + 1}`; };
const P = (s, opts = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 120 }, ...opts, children: rich(s) });
const H1 = (s, newPage = false) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: newPage, children: [new TextRun(s)] });
const PART = (s) => H1(s, true);   // starts a new page (whiteboard page plan)
const H2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(s)] });
const H3 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(s)] });
const bullet = (s) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { line: 336, after: 80 }, children: rich(s) });
const step = (n, s) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 567, hanging: 567 }, spacing: { line: 336, after: 80 }, children: [new TextRun({ text: `${n}.\t`, bold: true }), ...rich(s)] });
const eq = (s) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 140 }, children: rich(s, { italics: false, font: "Cambria Math" }) });

let figN = 0, tabN = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fig(key, file, widthIn, caption) {
  figN += 1;
  if (FIG_ORDER[figN - 1] !== key) throw new Error(`figure ${key} is out of order (expected ${FIG_ORDER[figN - 1]})`);
  const full = path.join(FIG, file);
  const { w, h } = pngSize(full);
  const width = Math.round(widthIn * 96);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width, height: Math.round(width * h / w) },
        altText: { title: `Figure ${figN}`, description: caption, name: file } })],
    }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figN}: `, bold: true, size: 20 }), ...rich(caption, { size: 20, italics: true })] }),
  ];
}
function tableCaption(key, caption) {
  tabN += 1;
  if (TAB_ORDER[tabN - 1] !== key) throw new Error(`table ${key} is out of order (expected ${TAB_ORDER[tabN - 1]})`);
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: `Table ${tabN}: `, bold: true, size: 20 }), ...rich(caption, { size: 20 })] });
}
function table(headers, rows, weights, opts = {}) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / total) * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, header, shade) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: header ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : shade ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ keepNext: true, alignment: i === 0 || opts.leftAll ? AlignmentType.LEFT : (opts.align || AlignmentType.CENTER),
      children: rich(String(text), { size: 20, bold: header || undefined }) })],
  });
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, opts.highlightRow === ri)) })),
    ],
  });
}
const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------- dataset preview ----------
const csvLines = fs.readFileSync(path.join(ROOT, cfg.data.path), "utf8").trim().split(/\r?\n/);
const preview = csvLines.slice(1, 6).map((l, i) => [String(i + 1), ...l.split(",")]);

// ======================================================================
// PAGE 1 — TITLE
// ======================================================================
const center = (text, size, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: opts.after ?? 120, before: opts.before ?? 0 },
  children: [new TextRun({ text, size, bold: opts.bold, italics: opts.italics, color: opts.color })] });
const titlePage = [
  center("PROJECT REPORT", 28, { bold: true, before: 1200, after: 600, color: GREY }),
  center("Neural Network Regression for Predicting", 40, { bold: true, after: 60 }),
  center("Chemical Reaction Yield from", 40, { bold: true, after: 60 }),
  center("Physicochemical Reaction Conditions", 40, { bold: true, after: 400 }),
  center("Project 9 — Industrial case study: Reliance Chemicals Ltd.", 26, { italics: true, after: 1400 }),
  center("Submitted by", 24, { color: GREY, after: 60 }),
  center("[Student Name]", 28, { bold: true, after: 40 }),
  center("Roll No. [Roll Number]", 24, { after: 600 }),
  center("Under the guidance of", 24, { color: GREY, after: 60 }),
  center("[Teacher / Supervisor Name]", 28, { bold: true, after: 40 }),
  center("[Designation], [Department]", 24, { after: 1400 }),
  center("[Department Name]", 24, { after: 40 }),
  center("[Institute Name]", 26, { bold: true, after: 40 }),
  center("October 2026", 24),
];

// ======================================================================
// PAGE 2 — ABSTRACT (300–400 words, 7-part structure from the whiteboard)
// ======================================================================
const abstract = [
  PART("Abstract"),
  P(`Chemical manufacturers such as Reliance Chemicals Ltd. depend on reaction yield — the share of the theoretically possible product that is actually obtained — for profitability, raw-material efficiency and energy use. Operating conditions that maximise yield are usually found through pilot trials or designed experiments, which are slow and expensive. Earlier work has approached yield prediction in three main ways: mechanistic kinetic models built from rate laws, response surface methodology (RSM) that fits low-order polynomials to designed experiments, and machine-learning models trained on the molecular structure of reactants and catalysts. These approaches leave clear gaps. Kinetic models require a fully characterised mechanism; quadratic RSM surfaces cannot follow sharp non-linear effects such as the loss of selectivity at high temperature; and most machine-learning studies predict yield from molecular identity rather than from the continuous conditions an operator actually controls.`),
  P(`This project addresses the last two gaps by building a model that maps continuous physicochemical conditions — temperature, pressure and reactant concentration — directly to yield without needing a mechanism. The method is a feed-forward neural network (YieldNet, 3–64–32–1, ReLU) implemented in PyTorch, with a single linear output neuron for regression, trained with mean squared error, the Adam optimiser and early stopping. Strict feature scaling is applied: the scaler is fitted on the training split only and reused unchanged for validation and test data, preventing information leakage. Because Reaxys data are licensed, ${ds.n_samples.toLocaleString("en-IN")} reactions were simulated from Arrhenius kinetics, a saturating pressure effect and a temperature-driven side reaction, with 2 % Gaussian measurement noise.`),
  P(`On a held-out test set of ${sz.test} reactions the network achieved R² = ${f(nn.r2, 3)}, MAE = ${f(nn.mae)} percentage points and RMSE = ${f(nn.rmse)}, close to the irreducible noise floor (RMSE ${f(ceil.rmse)}). It outperformed linear regression (R² = ${f(LIN.r2, 2)}), a quadratic RSM-style model (R² = ${f(POLY.r2, 2)}) and a random forest (R² = ${f(RF.r2, 3)}); five-fold cross-validation confirmed the result (R² = ${f(cv["Neural Network (YieldNet)"].r2.mean, 4)} ± ${f(cv["Neural Network (YieldNet)"].r2.std, 4)}). Residuals were centred on zero with constant spread and no trend. An ablation study showed that removing feature scaling reduced R² to ${f(ab["No feature scaling"].r2, 2)}, demonstrating that scaling is essential when inputs differ in magnitude. The model located the optimum reaction temperature within ${f(Math.abs(opt.true_T - opt.nn_T), 1)} K of the true value, showing its value as a fast screening tool before plant trials.`),
  new Paragraph({ spacing: { before: 200 }, children: rich("**Keywords:** reaction yield prediction, neural network regression, feature scaling, PyTorch, response surface methodology, residual analysis") }),
];

// ======================================================================
// PAGES 3–6 — INTRODUCTION
// ======================================================================
const intro = [
  PART("1. Introduction"),
  H2("1.1 Problem Statement"),
  P("Reaction yield is the amount of desired product obtained from a chemical reaction, expressed as a percentage of the maximum amount allowed by the stoichiometry. For a bulk chemical producer such as Reliance Chemicals Ltd., yield is directly linked to profitability: every percentage point of yield that is lost represents raw material that was bought, heated, pressurised and separated but did not become saleable product. Low yield also increases the load on downstream separation units and the volume of waste that has to be treated."),
  P("The yield of a given reaction depends strongly on the conditions under which it is run. Temperature controls the rate of the desired reaction through the Arrhenius law, but at high temperature side reactions and product decomposition also speed up. Pressure changes the concentration of gaseous reactants and can shift equilibria. Reactant concentration affects both the reaction rate and the balance between the main reaction and side reactions. These effects interact with one another, so the best combination of conditions cannot be found reliably by changing one variable at a time."),
  P("Today, operating conditions are typically selected through pilot-plant trials and designed experiments. Every trial consumes material, energy and time, so only a limited number can be run. A model that predicts yield from the reaction conditions would let engineers screen many candidate conditions on a computer and reserve physical trials for the most promising ones."),
  P("**Formal problem statement.** Given the continuous physicochemical conditions of a reaction — temperature *T* (K), pressure *P* (bar) and reactant concentration *C* (mol/L) — estimate the reaction yield *Y* (%). The task is to learn a function *f* such that *Ŷ = f(T, P, C) ≈ Y*, with *Y* in the range 0–100 %. Because yield is a continuous quantity and not a category, this is a **regression** problem, not a classification problem. The project brief requires the model to be a neural network."),
  P("*Note: Reliance Chemicals Ltd. is the industrial setting specified in the project brief. No proprietary company data were used; the dataset is simulated, as described in Section 2.1.*"),

  H2("1.2 Literature Review: How Others Have Solved the Problem"),
  P(`Published approaches to predicting or optimising reaction yield fall into four broad groups, summarised in ${T("lit")}.`),
  H3("1.2.1 Mechanistic kinetic models"),
  P("The classical chemical-engineering approach writes a rate equation for each elementary step, uses the Arrhenius equation *k = A·exp(−E_{a}/RT)* for the temperature dependence of each rate constant [4], and integrates the resulting differential equations over the reactor residence time [3]. Such models are interpretable and extrapolate well, but they need the full reaction mechanism together with the activation energy and pre-exponential factor of every step. For industrial reactions with several side reactions these parameters are often unknown and expensive to measure."),
  H3("1.2.2 Design of experiments and response surface methodology"),
  P("Response surface methodology (RSM), introduced by Box and Wilson [1], fits a low-order polynomial — usually quadratic — to yields measured at a designed set of conditions and then searches that surface for the optimum [2]. RSM is inexpensive, widely taught and needs no mechanism. Its main weakness is its fixed functional form: a quadratic surface has a single smooth, symmetric peak and cannot follow sharp changes such as the rapid loss of selectivity once a decomposition temperature is crossed."),
  H3("1.2.3 Machine learning on molecular structure"),
  P("Over the last decade, machine-learning models have been trained on large reaction datasets. Coley et al. [5] predicted the major products of organic reactions with a neural network. Ahneman et al. [6] used random forests on quantum-chemical descriptors to predict yields of Buchwald–Hartwig C–N couplings measured by high-throughput experimentation, and Schwaller et al. [8] fine-tuned a transformer language model on reaction SMILES strings to predict yield. Żurański et al. [10] review this field and note that model quality is limited by the size, consistency and noise of the available yield data. These models mainly answer *which reagents, catalyst or solvent* to use; the continuous process conditions are often held fixed or treated coarsely."),
  H3("1.2.4 High-throughput experimentation and Bayesian optimisation"),
  P("Automated platforms can now run thousands of small-scale reactions [7], and Bayesian optimisation chooses the next experiment so that an optimum is found in few trials [9]. These methods are powerful for locating an optimum, but they are sequential search procedures tied to a specific campaign; by themselves they do not deliver a reusable, fast model of yield over the whole operating window."),
  tableCaption("lit", "Summary of existing approaches to yield prediction"),
  table(["Approach", "Representative work", "How yield is predicted", "Limitation for this problem"], [
    ["Mechanistic kinetics", "Arrhenius [4]; Fogler [3]", "Rate laws integrated over residence time", "Needs complete mechanism and kinetic parameters"],
    ["DoE / RSM", "Box & Wilson [1]; Montgomery [2]", "Quadratic polynomial fitted to designed runs", "Rigid shape; misses sharp non-linear effects"],
    ["ML on molecules", "Ahneman [6]; Schwaller [8]; Żurański [10]", "Random forest / transformer on molecular descriptors", "Targets reagent choice, not continuous conditions"],
    ["HTE + Bayesian optimisation", "Perera [7]; Shields [9]", "Sequential search for the optimum", "Search procedure, not a reusable predictive model"],
  ], [2, 2.3, 2.6, 2.8], { leftAll: true }),
  spacer(),

  H2("1.3 Research Gaps"),
  bullet("**G1 — Mechanism dependence.** Kinetic models need a complete, parameterised mechanism, which is rarely available for industrial reactions."),
  bullet("**G2 — Rigid functional form.** Quadratic RSM surfaces cannot represent sharp non-linear effects such as selectivity loss at high temperature, so the predicted optimum can be wrong."),
  bullet("**G3 — Focus on molecular identity.** Most published machine-learning yield models predict from reactant and catalyst structures, not from the continuous conditions (*T*, *P*, *C*) that a plant operator adjusts day to day."),
  bullet("**G4 — Methodological rigour.** Small, noisy yield datasets make models sensitive to preprocessing; inconsistent feature scaling and leakage between training and test data can produce misleadingly optimistic results [10]."),
  bullet("**G5 — Data access.** Large curated databases such as Reaxys [18] are licensed, which limits reproducible academic work."),

  H2("1.4 Gap Addressed in This Work"),
  P("Out of these gaps, this project addresses **G2** and **G3**, supported by **G4**. It builds a neural-network regression model that maps the continuous conditions *T*, *P* and *C* directly to yield, needs no reaction mechanism, can follow sharp non-linear behaviour, and is trained and evaluated under a strict, leakage-free protocol. G5 is handled by a transparent, physically motivated simulator whose equations are given in this report, so that the study is fully reproducible. G1 is not solved directly; instead the neural network is offered as a data-driven alternative for situations where no mechanism is available."),

  H2("1.5 Motivation"),
  P("A feed-forward network with at least one hidden layer and a non-linear activation can approximate any continuous function on a bounded domain to arbitrary accuracy, given enough hidden units [11]. This makes it a natural choice when the true yield surface is non-linear and its mathematical form is unknown. Once trained, the network evaluates thousands of candidate conditions in milliseconds, which makes it practical as a screening tool ahead of costly plant trials. For Reliance Chemicals Ltd. this would mean fewer trial runs, faster identification of efficient operating windows, and less wasted raw material and energy."),

  H2("1.6 Objectives"),
  bullet("**O1.** Build a reproducible dataset of reaction conditions and yields with known, physically motivated structure."),
  bullet("**O2.** Design a preprocessing pipeline with strict feature scaling that prevents information leakage."),
  bullet("**O3.** Implement a neural network regression model (YieldNet) in PyTorch with an output layer suited to a continuous target."),
  bullet("**O4.** Evaluate the model with R², MAE, RMSE and residual analysis on an untouched test set and with 5-fold cross-validation."),
  bullet("**O5.** Compare the model with linear regression, quadratic RSM-style regression and a random forest."),
  bullet("**O6.** Quantify, through ablation, the contribution of feature scaling, of each input variable and of network size."),

  H2("1.7 Expected Outcomes"),
  bullet("A trained model that predicts yield to within a few percentage points, with error close to the measurement-noise level."),
  bullet("Evidence that the neural network captures non-linear effects that linear and quadratic models miss."),
  bullet("A quantitative demonstration of why feature scaling is necessary when inputs have different units and magnitudes."),
  bullet("A reusable code base (data generation, training, evaluation, prediction) and a residual analysis that supports the reliability of the model."),
];

// ======================================================================
// PAGES 7–9 — METHODOLOGY
// ======================================================================
const methodology = [
  PART("2. Methodology"),
  H2("2.1 Dataset"),
  H3("2.1.1 Source"),
  P(`The project brief permits data either from Reaxys or from simulation. Reaxys [18] is a commercial database licensed by Elsevier and was not available for this work, so a physically motivated simulator was used. The simulator is described in full below, so every result in this report can be reproduced exactly with random seed ${cfg.seed}.`),
  H3("2.1.2 What the data contain"),
  P(`Each record is one reaction run, described by three input variables and one target (${T("vars")}). Input values were drawn independently and uniformly over ranges typical of a liquid-phase reaction under gas pressure, which mimics a space-filling experimental design.`),
  tableCaption("vars", "Variables in the dataset"),
  table(["Variable", "Symbol", "Unit", "Range", "Role"], [
    ["Temperature", "T", "K", "300 – 450", "Input"],
    ["Pressure", "P", "bar", "1 – 50", "Input"],
    ["Reactant concentration", "C", "mol/L", "0.1 – 2.0", "Input"],
    ["Reaction yield", "Y", "%", "0 – 100", "Target (continuous)"],
  ], [2.6, 1.2, 1.2, 1.6, 2]),
  spacer(),
  P("Yield is generated as the product of conversion *X* and selectivity *S*, plus measurement noise *ε*:"),
  eq("Y = 100 · X(T, P, C) · S(T, C) + ε,     ε ~ N(0, 2²)"),
  eq("X = 1 − exp[ −k(T) · P/(P + K_{P}) · √C ],     k(T) = A · exp(−E_{a} / RT)"),
  eq("S = 1 / {1 + exp[(T − T_{d}) / w]}  ·  1 / (1 + βC²)"),
  P("with *E_{a}* = 60 kJ/mol, *A* = 3.4 × 10^{8} (pre-exponential factor multiplied by a fixed reaction time), *K_{P}* = 5 bar, *T_{d}* = 420 K, *w* = 12 K and *β* = 0.1. Conversion rises with temperature (Arrhenius kinetics), saturates with pressure and grows with concentration. Selectivity collapses above about 420 K as a side reaction takes over, and falls slightly at high concentration. Together these give an optimum temperature window and strongly non-linear interactions. Yields were clipped to 0–100 % after noise was added; the noise level of 2 percentage points represents typical analytical measurement error and sets a floor on how accurate any model can be."),

  H2("2.2 Data Preprocessing"),
  P("Preprocessing follows a fixed order, so that no information from the validation or test data reaches the model during training."),
  step(1, `**Integrity checks, then split.** The dataset has ${ds.missing_values} missing values and every variable lies inside its stated range, so no imputation or outlier removal was needed. The records are then randomly divided into training (70 %, ${sz.train.toLocaleString("en-IN")} rows), validation (15 %, ${sz.val} rows) and test (15 %, ${sz.test} rows) sets with seed ${cfg.seed}, before any statistics are computed.`),
  step(2, `**Strict feature scaling.** Each input is standardised as *z = (x − μ_{train}) / σ_{train}*, with mean *μ* and standard deviation *σ* computed on the training set only (${T("scaler")}). The same *μ* and *σ* are applied unchanged to the validation and test sets and to every future prediction, and they are saved with the model weights.`),
  step(3, `**Target standardisation.** For training, yield is also standardised with training-set statistics (mean ${f(sc.target_mean)} %, standard deviation ${f(sc.target_std)} %). Predictions are converted back to percent before any metric is computed, so all reported errors are in yield percentage points.`),
  tableCaption("scaler", "Scaling parameters, fitted on the training set only"),
  table(["Feature", "Training mean μ", "Training std. dev. σ"], [
    ["Temperature (K)", f(sc.mean.temperature_K), f(sc.std.temperature_K)],
    ["Pressure (bar)", f(sc.mean.pressure_bar), f(sc.std.pressure_bar)],
    ["Concentration (mol/L)", f(sc.mean.concentration_molL, 4), f(sc.std.concentration_molL, 4)],
  ], [3, 2, 2]),
  spacer(),
  P(`**Why scaling is crucial.** In the training data, temperature averages about ${f(sc.mean.temperature_K, 0)} K, while concentration averages about ${f(sc.mean.concentration_molL, 1)} mol/L (${F("scaling")}). The gradient of the loss with respect to a first-layer weight is proportional to the input value attached to it, so without scaling, a weight on temperature receives a gradient roughly ${Math.round(sc.mean.temperature_K / sc.mean.concentration_molL)} times larger than a weight on concentration, so one learning rate cannot suit both, and the large raw values push ReLU units into poorly conditioned activations [12]. Scaling gives every input mean 0 and standard deviation 1, so the network learns from the information in each variable, not its unit. Fitting the scaler on training data only makes the scaling *strict*: test-set statistics would leak test information into training and inflate the reported accuracy.`),

  H2("2.3 Algorithm: YieldNet Neural Network Regression"),
  P(`${F("flowchart")} shows the complete pipeline. YieldNet itself is a fully connected feed-forward network with three inputs, two hidden layers of 64 and 32 units with ReLU activation, and a single output unit. For a standardised input vector *z*:`),
  ...fig("flowchart", "fig01_pipeline_flowchart.png", 5.0, "Flow chart of the pipeline. Blue boxes mark the core components: strict scaling and the YieldNet regression network."),
  eq("h_{1} = ReLU(W_{1}z + b_{1}),     h_{2} = ReLU(W_{2}h_{1} + b_{2}),     ŷ = w_{3}^{T}h_{2} + b_{3}"),
  P(`The network has ${nn.parameters.toLocaleString("en-IN")} trainable parameters, learned by minimising the mean squared error between predicted and actual standardised yield:`),
  eq("L = (1/n) · Σ_{i} (y_{i} − ŷ_{i})²"),
  P(`**Output layer for regression.** The final layer is one neuron with no activation (identity), so its output is an unbounded real number read directly as the predicted yield. A classifier would end in softmax (or sigmoid) neurons that output class probabilities and would be trained with cross-entropy (${T("output")}).`),
  tableCaption("output", "Output layer: classification versus regression (this project)"),
  table(["Aspect", "Classification", "Regression (YieldNet)"], [
    ["Output neurons", "One per class (K)", "One"],
    ["Output activation", "Softmax (or sigmoid)", "None (identity / linear)"],
    ["Output meaning", "Class probabilities, sum to 1", "Predicted yield, a real number"],
    ["Loss function", "Cross-entropy", "Mean squared error"],
    ["Evaluation", "Accuracy, F1, confusion matrix", "R², MAE, RMSE, residual plot"],
  ], [2.2, 3, 3]),
  spacer(),
  P("**Training procedure (Algorithm 1).**"),
  step(1, `Fix the random seed (${cfg.seed}) and initialise the weights; each epoch, shuffle the training set into mini-batches of ${cfg.training.batch_size}.`),
  step(2, `For each mini-batch: forward pass, compute MSE, back-propagate, and update the weights with Adam [14] (learning rate ${cfg.training.learning_rate}, weight decay 10^{−5}).`),
  step(3, `After each epoch, compute the validation MSE and keep the best weights; stop after ${cfg.training.patience} epochs without improvement (early stopping [13]) or ${cfg.training.max_epochs} epochs, restore the best weights, then predict on the test set and compute the metrics in percent.`),
];

// ======================================================================
// PAGES 10–18 — EXPERIMENTS AND RESULTS
// ======================================================================
const nnCv = cv["Neural Network (YieldNet)"];
const cvRow = (name, label) => [label, `${f(cv[name].r2.mean, 4)} ± ${f(cv[name].r2.std, 4)}`, `${f(cv[name].mae.mean)} ± ${f(cv[name].mae.std)}`, `${f(cv[name].rmse.mean)} ± ${f(cv[name].rmse.std)}`];
const abRow = (key, change) => [key, change, f(ab[key].r2, 4), f(ab[key].mae), f(ab[key].rmse), (ab[key].r2 - nn.r2 >= 0 ? "+" : "") + f(ab[key].r2 - nn.r2, 4)];
const bands = res.by_predicted_band;

const experiments = [
  PART("3. Experiments and Results"),
  H2("3.1 Experimental Setup (Configuration)"),
  P(`All settings are stored in one configuration file (configs/config.json), and every experiment is run by one script (run_experiments.py), so the whole study can be repeated with a single command. ${T("config")} lists the configuration.`),
  tableCaption("config", "Experimental configuration"),
  table(["Parameter", "Value"], [
    ["Random seed", String(cfg.seed)],
    ["Samples", `${ds.n_samples.toLocaleString("en-IN")} (simulated)`],
    ["Train / validation / test split", `70 / 15 / 15 % (${sz.train.toLocaleString("en-IN")} / ${sz.val} / ${sz.test} rows)`],
    ["Feature scaling", "StandardScaler, fitted on training set only"],
    ["Architecture", "3 → 64 → 32 → 1, ReLU hidden layers, linear output"],
    ["Trainable parameters", nn.parameters.toLocaleString("en-IN")],
    ["Loss / optimiser", `MSE / Adam, learning rate ${cfg.training.learning_rate}, weight decay 10^{−5}, batch size ${cfg.training.batch_size}`],
    ["Epochs: maximum / patience / run (best)", `${cfg.training.max_epochs} / ${cfg.training.patience} / ${nn.epochs_run} (${nn.best_epoch})`],
    ["Cross-validation", `${cfg.cv_folds}-fold on the ${(sz.train + sz.val).toLocaleString("en-IN")}-row development set`],
    ["Software", `Python 3.11, PyTorch ${M.environment.torch}, scikit-learn 1.5.2, NumPy ${M.environment.numpy}, pandas ${M.environment.pandas}`],
    ["Hardware / training time", `CPU only (Windows 11); ${f(nn.train_seconds, 0)} s for the final model`],
  ], [3, 5], { leftAll: true }),
  spacer(),

  H2("3.2 Comparison Methods"),
  P("Three baselines were trained on the same training split and evaluated on the same test split:"),
  bullet("**Linear regression** on standardised inputs — the simplest model, which tests whether the problem needs non-linearity at all."),
  bullet("**Polynomial regression of degree 2** — a full quadratic model with interaction terms, equivalent to the second-order model used in response surface methodology [1, 2]; this represents the standard industrial approach."),
  bullet(`**Random forest** with ${cfg.baselines.random_forest_trees} trees [15] — a strong non-linear, non-parametric model widely used for tabular data and for yield prediction [6].`),
  P("The linear and polynomial models were implemented as scikit-learn [17] pipelines that include a standard scaler fitted on the training data. The random forest was given unscaled inputs, because tree splits do not depend on the scale of a feature. The neural network was implemented in PyTorch [16]."),

  H2("3.3 Datasets"),
  P(`The experiments use the simulated dataset described in Section 2.1, split 70 / 15 / 15 % as listed in ${T("config")}: the training set fits the scalers and weights, the validation set chooses the best epoch, and the test set is used once for the final scores. ${T("preview")} shows the first five records. ${F("dist")} shows the distribution of each variable. The three inputs are spread evenly across their ranges, as expected from uniform sampling. The yield is right-skewed: its mean is ${f(ds.summary.yield_pct.mean, 1)} %, its median is ${f(ds.summary.yield_pct["50%"], 1)} %, its maximum is ${f(ds.summary.yield_pct.max, 1)} %, and ${f(ds.share_yield_below_10pct * 100, 1)} % of runs give less than 10 % yield. Most of these low-yield runs are at low temperature, where the reaction is too slow to reach high conversion in the available time.`),
  tableCaption("preview", `First five records of the dataset (${ds.n_samples.toLocaleString("en-IN")} rows in total)`),
  table(["#", "Temperature (K)", "Pressure (bar)", "Concentration (mol/L)", "Yield (%)"], preview, [0.6, 2, 2, 2.4, 1.6]),
  ...fig("dist", "fig02_distributions.png", 4.2, "Distributions of the three reaction conditions and of the yield."),
  P(`${F("scaling")} shows the same training-set inputs before and after the strict scaling of Section 2.2: raw temperature values are hundreds of times larger than concentration values, while after standardisation all three share the same range.`),
  ...fig("scaling", "fig05_scaling_effect.png", 5.0, "Training-set features before scaling (left) and after standardisation (right)."),
  P(`${F("yvf")} plots yield against each condition. Temperature shows a clear rise-and-fall shape with a peak near 390–400 K. Pressure shows a weak, saturating upward trend. Concentration shows no visible trend on its own, because its effect is masked by the much larger temperature effect.`),
  ...fig("yvf", "fig04_yield_vs_features.png", 5.6, "Yield plotted against each reaction condition (all 2,000 runs)."),
  P(`The linear (Pearson) correlations confirm this picture: yield correlates with temperature at r = ${f(ds.correlation_with_yield.temperature_K)}, with pressure at r = ${f(ds.correlation_with_yield.pressure_bar)} and with concentration at r = ${f(ds.correlation_with_yield.concentration_molL)}. The inputs are uncorrelated with each other, so there is no multicollinearity. The low correlations do *not* mean that pressure and concentration are unimportant: correlation only measures straight-line relationships, and the ablation study in Section 3.5.3 shows that both variables carry useful non-linear information. The low correlations do, however, predict that a linear model will perform poorly.`),

  H2("3.4 Evaluation Protocol"),
  bullet("**Hold-out test.** The test set was used only once, to produce the final scores. All design choices were fixed in the configuration file before testing."),
  bullet("**Metrics.** Three standard regression metrics are reported, where *y_{i}* is the actual yield, *ŷ_{i}* the prediction and *ȳ* the mean actual yield:"),
  eq("R² = 1 − Σ(y_{i} − ŷ_{i})² / Σ(y_{i} − ȳ)²"),
  eq("MAE = (1/n) · Σ|y_{i} − ŷ_{i}|          RMSE = √[ (1/n) · Σ(y_{i} − ŷ_{i})² ]"),
  P("R² is the fraction of yield variance explained by the model (1 is perfect, 0 is no better than always predicting the mean). MAE is the average error in yield percentage points, which is the most direct figure for a plant engineer. RMSE penalises large errors more heavily than MAE."),
  bullet("**Cross-validation.** 5-fold cross-validation on the development set. In each fold, scalers are refitted on that fold's training data, and the network uses 15 % of the fold's training data for early stopping."),
  bullet("**Residual analysis.** Residual = actual − predicted. Residuals are plotted against the predicted value and checked for zero mean, constant spread and absence of pattern."),
  bullet(`**Noise ceiling.** Because the simulator is known, the noise-free function itself can be scored against the noisy test labels. Its scores (R² = ${f(ceil.r2, 4)}, MAE = ${f(ceil.mae)}, RMSE = ${f(ceil.rmse)}) are the best that any model could achieve; a model close to this ceiling has learned essentially all of the learnable structure.`),

  H2("3.5 Results"),
  H3("3.5.1 Quantitative results"),
  P(`${T("test")} and ${F("comparison")} compare all four models on the test set. The neural network is best on all three metrics, with R² = ${f(nn.r2, 4)}, MAE = ${f(nn.mae)} and RMSE = ${f(nn.rmse)} percentage points. Its RMSE is only ${rmseGapPct} % above the noise ceiling of ${f(ceil.rmse)}, so almost all of the remaining error is measurement noise that no model can remove. The random forest comes a close second (R² = ${f(RF.r2, 4)}), but its MAE is ${rfMaeGapPct} % higher. The quadratic RSM-style model explains only ${f(POLY.r2 * 100, 0)} % of the variance, with an average error of ${f(POLY.mae, 1)} points, and linear regression explains only ${f(LIN.r2 * 100, 0)} %.`),
  tableCaption("test", "Test-set performance of all models (300 unseen reactions); best values in bold"),
  table(["Model", "R²", "MAE (%)", "RMSE (%)", "Training time (s)"], [
    ["**Neural network (YieldNet)**", `**${f(nn.r2, 4)}**`, `**${f(nn.mae)}**`, `**${f(nn.rmse)}**`, f(nn.train_seconds, 1)],
    ["Random forest", f(RF.r2, 4), f(RF.mae), f(RF.rmse), f(RF.train_seconds, 2)],
    ["Polynomial regression (deg 2)", f(POLY.r2, 4), f(POLY.mae), f(POLY.rmse), f(POLY.train_seconds, 2)],
    ["Linear regression", f(LIN.r2, 4), f(LIN.mae), f(LIN.rmse), f(LIN.train_seconds, 2)],
    ["*Noise ceiling (true function)*", `*${f(ceil.r2, 4)}*`, `*${f(ceil.mae)}*`, `*${f(ceil.rmse)}*`, "—"],
  ], [3.4, 1.2, 1.2, 1.2, 1.6], { highlightRow: 0 }),
  ...fig("comparison", "fig09_model_comparison.png", 5.6, "Test-set R² (left) and MAE (right) for each model. The neural network is highlighted."),
  P(`Cross-validation (${T("cv")}) confirms the hold-out result. The neural network reaches R² = ${f(nnCv.r2.mean, 4)} with a standard deviation of only ${f(nnCv.r2.std, 4)} across the five folds, so its performance does not depend on a lucky split. The ranking of the four models is the same in every metric.`),
  tableCaption("cv", "5-fold cross-validation on the development set (mean ± standard deviation)"),
  table(["Model", "R²", "MAE (%)", "RMSE (%)"], [
    cvRow("Neural Network (YieldNet)", "Neural network (YieldNet)"),
    cvRow("Random Forest", "Random forest"),
    cvRow("Polynomial Regression (deg 2, RSM-style)", "Polynomial regression (deg 2)"),
    cvRow("Linear Regression", "Linear regression"),
  ], [3.4, 2, 2, 2]),
  spacer(),

  H3("3.5.2 Qualitative results"),
  P(`**Training behaviour.** ${F("trainfit")}(a) shows the training and validation loss. Both fall steeply during the first 25 epochs and then flatten. The validation loss stays close to the training loss throughout, so the network is not overfitting. The best validation loss was reached at epoch ${nn.best_epoch}; training stopped at epoch ${nn.epochs_run}, after ${cfg.training.patience} epochs without improvement, and the best weights were restored.`),
  P(`**Predicted versus actual.** In ${F("trainfit")}(b) the test predictions lie tightly along *y = x* from 0 to about 80 %, with no bending at either end. The linear output can dip slightly below 0 % for very slow reactions (lowest ${f(res.pred_min)} %, within the noise), so the prediction script clips outputs to 0–100 %.`),
  ...fig("trainfit", "fig13_training_and_fit.png", 5.6, "(a) Training and validation loss (MSE on the standardised target, log scale); the dashed line marks the epoch with the lowest validation loss. (b) Predicted versus actual yield on the test set; the dashed line is perfect prediction."),
  P(`**Residual plot.** A residual is the actual yield minus the predicted yield. When residuals are plotted against the predicted value (${F("residuals")}, left), a good regression model shows a structureless horizontal band centred on zero. Three properties were checked:`),
  bullet(`**Centred on zero.** The mean residual is ${f(res.mean)} points, essentially zero, so the model is not systematically over- or under-predicting.`),
  bullet(`**Constant spread (homoscedasticity).** The correlation between the size of a residual and the predicted value is only ${f(res.corr_abs_resid_vs_pred)}, and the residual standard deviation is about the same in every yield band (${T("bands")}). The model is equally reliable for low-yield and high-yield conditions.`),
  bullet(`**No pattern.** There is no curve or funnel shape, which means the non-linear structure of the data has been captured. ${f(res.share_within_2std * 100, 0)} % of residuals lie inside ±2σ, close to the 95 % expected for normally distributed errors.`),
  P(`The histogram (${F("residuals")}, right) closely matches the N(0, 2²) curve of the simulated measurement noise, with a standard deviation of ${f(res.std)} against the true noise level of ${f(res.noise_floor_std, 1)}. The residuals are therefore almost pure noise. The one small effect visible in ${T("bands")} is a mean residual of +${f(bands["> 60 %"].mean)} points for predictions above 60 %: the model slightly under-predicts near the yield peak, where the surface is most sharply curved.`),
  ...fig("residuals", "fig08_residuals.png", 5.6, "Residual analysis on the test set: residuals against predicted yield with a ±2σ band (left), and the residual histogram compared with the simulated noise distribution (right)."),
  tableCaption("bands", "Residuals by predicted-yield band (test set)"),
  table(["Predicted yield band", "Test reactions", "Mean residual (%)", "Std. dev. of residual (%)"],
    Object.entries(bands).map(([k, v]) => [k, String(v.n), (v.mean >= 0 ? "+" : "") + f(v.mean), f(v.std)]), [2.4, 1.6, 2, 2.4]),
  spacer(),
  P(`${F("contrast")} shows why the residual plot matters. The quadratic RSM-style model leaves a strong U-shaped pattern in its residuals: it over-predicts mid-range yields and under-predicts both ends. This pattern is the classic sign of underfitting — the model's shape is too simple for the data. The quadratic model even predicts physically impossible negative yields (as low as ${f(opt.poly_min_test_pred, 1)} %). The neural network's residuals, on the same scale, form a narrow random band.`),
  ...fig("contrast", "fig12_residual_contrast.png", 5.6, "Residual plots of the quadratic RSM-style model (left) and the neural network (right) on the same vertical scale."),
  P(`**Response curves.** ${F("response")} varies one condition at a time and compares each model with the true simulator. Along temperature, the neural network follows the asymmetric peak closely and places the optimum at ${f(opt.nn_T, 1)} K (${f(opt.nn_yield, 1)} % yield), against a true optimum of ${f(opt.true_T, 1)} K (${f(opt.true_yield, 1)} %). The quadratic model can only draw a symmetric parabola: it places the optimum at ${f(opt.poly_T, 1)} K and underestimates the achievable yield by about ${f(opt.true_yield - opt.poly_yield, 0)} points (${f(opt.poly_yield, 1)} %). The random forest follows the true curve but in a jagged, step-wise way, because each tree predicts a constant value within each region of the input space. The network gives a smooth curve, which is better suited to finding an optimum.`),
  ...fig("response", "fig10_response_curves.png", 5.3, "One-factor-at-a-time response curves: predicted yield against temperature (left) and pressure (right), compared with the true simulator (dashed)."),

  H3("3.5.3 Ablation study"),
  P(`Each ablation changes exactly one thing relative to the full model and keeps everything else — data split, optimiser, learning rate, epoch budget and seed — identical (${T("ablation")}).`),
  tableCaption("ablation", "Ablation study (test set); ΔR² is relative to the full model"),
  table(["Variant", "What changes", "R²", "MAE (%)", "RMSE (%)", "ΔR²"], [
    ["Full model", "—", f(nn.r2, 4), f(nn.mae), f(nn.rmse), "—"],
    abRow("No feature scaling", "Raw inputs (no StandardScaler)"),
    abRow("Without temperature", "Input T removed"),
    abRow("Without pressure", "Input P removed"),
    abRow("Without concentration", "Input C removed"),
    abRow("Architecture 32", `One hidden layer (${ab["Architecture 32"].parameters} params)`),
    abRow("Architecture 128-64-32", `Three hidden layers (${ab["Architecture 128-64-32"].parameters.toLocaleString("en-IN")} params)`),
  ], [2.3, 2.9, 1.1, 1.1, 1.1, 1.1]),
  spacer(),
  P(`**Feature scaling.** Removing feature scaling is the most damaging change that keeps all inputs: R² falls from ${f(nn.r2, 3)} to ${f(ab["No feature scaling"].r2, 3)} and MAE rises from ${f(nn.mae)} to ${f(ab["No feature scaling"].mae)} points. ${F("ablation")} shows why. With raw inputs, the validation loss drops once and then stalls near 1.0 (on the standardised target, a loss of 1.0 is no better than always predicting the mean yield). With scaling, the validation loss falls about a hundred times lower. The architecture, optimiser and number of epochs are identical in both runs; only the scale of the inputs differs.`),
  ...fig("ablation", "fig11_scaling_ablation.png", 3.5, "Validation loss with and without feature scaling (log scale)."),
  P(`**Input variables.** Removing temperature destroys the model (R² = ${f(ab["Without temperature"].r2, 3)}, no better than predicting the mean), which confirms that temperature is the dominant variable. Removing concentration (R² = ${f(ab["Without concentration"].r2, 3)}) hurts more than removing pressure (R² = ${f(ab["Without pressure"].r2, 3)}), even though concentration has almost zero linear correlation with yield (r = ${f(ds.correlation_with_yield.concentration_molL)}). The network therefore extracts non-linear information that a correlation analysis would miss, and all three inputs are worth measuring.`),
  P(`**Network size.** A single hidden layer of 32 units (${ab["Architecture 32"].parameters} parameters) reaches R² = ${f(ab["Architecture 32"].r2, 4)} but needs the full ${ab["Architecture 32"].epochs_run}-epoch budget. A deeper 128–64–32 network (${ab["Architecture 128-64-32"].parameters.toLocaleString("en-IN")} parameters) reaches R² = ${f(ab["Architecture 128-64-32"].r2, 4)}, only ${f(ab["Architecture 128-64-32"].r2 - nn.r2, 4)} better with ${f(ab["Architecture 128-64-32"].parameters / nn.parameters, 1)}× the parameters, so 64–32 is a good balance.`),

  H2("3.6 Discussion"),
  P("**Why the neural network beats RSM.** Yield here is shaped by competing exponential effects: Arrhenius acceleration on one side and selectivity collapse on the other. A parabola cannot represent this asymmetric, sharp-sided peak, so an RSM study would misplace the optimum, underestimate the achievable yield and could recommend the wrong operating point. The network learns the shape from data without assuming a form."),
  P("**Neural network versus random forest.** The random forest is nearly as accurate, as expected for a strong tabular model. The network is preferred because it is more accurate on every metric, its smooth, differentiable surface suits gradient-based or Bayesian search for the best conditions, and it is compact (about 2,400 parameters against 300 full trees)."),
  P("**Limitations.** (i) The data are simulated; real plant data have larger, non-Gaussian noise, sensor drift and unrecorded variables, so real accuracy will be lower. (ii) Only three conditions are modelled. (iii) The model is valid only inside its training ranges. (iv) It gives no uncertainty estimate."),
];

// ======================================================================
// PAGES 19–20 — CONCLUSION, FUTURE WORK, REFERENCES
// ======================================================================
const conclusion = [
  PART("4. Conclusion"),
  P(`This project developed a neural network regression model that predicts chemical reaction yield from three continuous physicochemical conditions: temperature, pressure and reactant concentration. A three-layer PyTorch network with a single linear output neuron, trained on strictly scaled inputs with mean squared error, Adam and early stopping, achieved R² = ${f(nn.r2, 3)} and MAE = ${f(nn.mae)} percentage points on unseen reactions, close to the irreducible measurement noise. It clearly outperformed linear regression and the quadratic model used in response surface methodology, and slightly outperformed a random forest while giving a smoother, more useful prediction surface. The residual analysis showed unbiased, constant-variance errors with no remaining structure. The ablation study showed that strict feature scaling is essential (R² ${f(nn.r2, 2)} with scaling against ${f(ab["No feature scaling"].r2, 2)} without) and that all three conditions contribute to the prediction. For Reliance Chemicals Ltd., such a model can serve as a fast screening tool that narrows the set of conditions to test in the plant, saving material, energy and time.`),
  H1("5. Future Work Plan"),
  tableCaption("future", "Planned extensions"),
  table(["Phase", "Work", "Expected benefit"], [
    ["1. Real data", "Obtain Reaxys data through an institutional licence and historical batch records; retrain and re-evaluate the same pipeline", "Validates the approach on real noise and real chemistry"],
    ["2. More conditions", "Add residence time, catalyst loading, solvent and stirring rate; encode categorical variables", "Wider coverage of what operators control"],
    ["3. Uncertainty", "Deep ensembles [20] or Monte Carlo dropout [21] to output prediction intervals", "Engineers know when to trust a prediction"],
    ["4. Physics-informed model", "Add Arrhenius structure and 0–100 % bounds into the network or its loss [19]", "Better extrapolation and fewer data needed"],
    ["5. Optimisation", "Use YieldNet as a surrogate inside Bayesian optimisation [9] to propose the next plant trials", "Fewer trials to reach the optimum"],
    ["6. Deployment", "Simple web or API tool with input-range checks and logging", "Daily use by process engineers"],
  ], [1.8, 4.2, 3], { leftAll: true }),
  spacer(),
  H1("References"),
  ...[
    "Box, G. E. P., & Wilson, K. B. (1951). On the experimental attainment of optimum conditions. *Journal of the Royal Statistical Society: Series B*, 13(1), 1–45.",
    "Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley.",
    "Fogler, H. S. (2016). *Elements of Chemical Reaction Engineering* (5th ed.). Prentice Hall.",
    "Arrhenius, S. (1889). Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren. *Zeitschrift für Physikalische Chemie*, 4, 226–248.",
    "Coley, C. W., Barzilay, R., Jaakkola, T. S., Green, W. H., & Jensen, K. F. (2017). Prediction of organic reaction outcomes using machine learning. *ACS Central Science*, 3(5), 434–443.",
    "Ahneman, D. T., Estrada, J. G., Lin, S., Dreher, S. D., & Doyle, A. G. (2018). Predicting reaction performance in C–N cross-coupling using machine learning. *Science*, 360(6385), 186–190.",
    "Perera, D., Tucker, J. W., Brahmbhatt, S., Helal, C. J., Chong, A., Farrell, W., Richardson, P., & Sach, N. W. (2018). A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. *Science*, 359(6374), 429–434.",
    "Schwaller, P., Vaucher, A. C., Laino, T., & Reymond, J.-L. (2021). Prediction of chemical reaction yields using deep learning. *Machine Learning: Science and Technology*, 2(1), 015016.",
    "Shields, B. J., Stevens, J., Li, J., Parasram, M., Damani, F., Martinez Alvarado, J. I., Janey, J. M., Adams, R. P., & Doyle, A. G. (2021). Bayesian reaction optimization as a tool for chemical synthesis. *Nature*, 590, 89–96.",
    "Żurański, A. M., Martinez Alvarado, J. I., Shields, B. J., & Doyle, A. G. (2021). Predicting reaction yields via supervised learning. *Accounts of Chemical Research*, 54(8), 1856–1865.",
    "Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.",
    "LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp. In G. B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade* (pp. 9–50). Springer.",
    "Prechelt, L. (1998). Early stopping — but when? In G. B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade* (pp. 55–69). Springer.",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.",
    "Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.",
    "Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.",
    "Elsevier. (n.d.). *Reaxys* [Chemistry database]. https://www.reaxys.com",
    "Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.",
    "Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30.",
    "Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, 1050–1059.",
  ].map((r, i) => new Paragraph({ indent: { left: 567, hanging: 567 }, spacing: { after: 80, line: 276 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: `[${i + 1}]\t`, size: 22 }), ...rich(r, { size: 22 })] })),
];

const appendix = [
  PART("Appendix A: Reproducing the Results"),
  P("All code, data and figures are produced by the project folder below. With Python 3.11 installed, run the following from the project folder:"),
  ...["pip install -r requirements.txt", "npm install", "python run_experiments.py", "python predict.py --temperature 392 --pressure 25 --concentration 1.0", "node report/build_report.js"]
    .map((c) => new Paragraph({ spacing: { after: 40 }, indent: { left: 567 }, children: [new TextRun({ text: c, font: "Consolas", size: 20 })] })),
  P("The first two commands install the Python and Node.js dependencies. The next regenerates the dataset, trains every model, runs the ablations and cross-validation, and writes results/metrics.json and all figures (about two minutes on a laptop CPU). The fourth predicts the yield for new conditions with the saved model. The last rebuilds this report from the saved results, so every number in the report matches the latest run.", { spacing: { before: 120, line: 360, after: 120 } }),
  tableCaption("structure", "Project structure"),
  table(["Path", "Contents"], [
    ["configs/config.json", "All hyperparameters and settings"],
    ["src/data_generator.py", "Reaction simulator (Section 2.1)"],
    ["src/preprocess.py", "Split and strict feature scaling (Section 2.2)"],
    ["src/model.py", "YieldNet architecture (Section 2.3)"],
    ["src/train.py", "Training loop with Adam and early stopping"],
    ["src/metrics.py, src/plots.py", "Metrics and all figures"],
    ["run_experiments.py", "Runs the full study end to end"],
    ["predict.py", "Command-line yield prediction"],
    ["data/reaction_yield.csv", "The 2,000-row dataset"],
    ["results/", "metrics.json, comparison.csv, ablation.csv, test_predictions.csv, figures/, model/"],
  ], [3, 5], { leftAll: true }),
];

// ---------- document ----------
const headerText = "Reaction Yield Prediction using Neural Network Regression";
const doc = new Document({
  creator: "[Student Name]",
  title: "Neural Network Regression for Predicting Chemical Reaction Yield",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] }],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } }, titlePage: true },
    headers: {
      first: new Header({ children: [new Paragraph({ children: [] })] }),
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB", space: 4 } },
        children: [new TextRun({ text: headerText, size: 18, italics: true, color: GREY })] })] }),
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }),
    },
    children: [
      ...titlePage, ...abstract, ...intro, ...methodology, ...experiments, ...conclusion, ...appendix,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`Wrote ${OUT} (${figN} figures, ${tabN} tables)`);
});
