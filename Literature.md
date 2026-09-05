# Reading Map — Electricity Price Formation, Forecasting & Interpretability

*A territory map, not a route. Everything below is here so you can see what exists and decide for yourself what deserves your six months. Where the field genuinely disagrees, I've given you both sides and stayed out of it.*

---

## 0. How to use this

**Read the debates, not the conclusions.** Nearly every section below has a "⚔️ Live disagreement" marker. Those are the places where competent researchers currently contradict each other. They are where your own judgment has room to operate, and where a thesis can say something rather than just report something.

**On source quality.** You'll notice a lot of MDPI journals (*Energies*, *Applied Sciences*, *MAKE*) in this field. They're fast, open-access, and enormously variable — some are excellent, some are recycled. Similarly, arXiv preprints in EPF range from field-defining to unreviewed. Develop a habit early: check whether a paper (a) benchmarks against a simple model, (b) tests significance, (c) states its information set. Papers failing all three are common and can be skimmed for ideas rather than trusted for numbers. The Weron/Lago/Ziel school in *International Journal of Forecasting*, *Applied Energy*, *Energy Economics* and *IEEE Trans. Power Systems* is the methodological centre of gravity; use it as your calibration.

**How to read papers efficiently:** S. Keshav, *"How to Read a Paper"* (2007, 2 pages, free). Three-pass method. Worth the ten minutes before you read 80 papers.

**Suggested rhythm for 6 months:** roughly 6–8 weeks of broad orientation (Parts 1–3 + skimming 4 and 5), then narrowing. Don't try to finish this list. It's designed to be over-complete so that you can see the shape of the field and pick.

---

## PART 1 — Electricity markets and power system economics

*The domain layer. If you skip this, you'll build a good regression and have nothing to say about it. This is also where an Energy Engineering degree gives you an advantage over a pure ML student — use it.*

### Books (pick one or two, don't read all)

- **Kirschen & Strbac, _Fundamentals of Power System Economics_** (Wiley, 2nd ed. 2018). The standard engineering-school text. Marginal pricing, merit order, congestion, market clearing. Probably the most directly aligned with your background.
- **Biggar & Hesamzadeh, _The Economics of Electricity Markets_** (Wiley, 2014). More rigorous economics, less engineering. Excellent on why marginal pricing is the design it is.
- **Stoft, _Power System Economics: Designing Markets for Electricity_** (IEEE/Wiley, 2002). Older, more conceptual, famously readable. Great on the *logic* of market design — the "why" behind rules that otherwise look arbitrary.
- **Harris, _Electricity Markets: Pricing, Structures and Economics_** (Wiley, 2006). Practitioner-flavoured, trading-desk perspective.
- **Weron, _Modeling and Forecasting Electricity Loads and Prices: A Statistical Approach_** (Wiley, 2006). Older but foundational; the statistical stylised facts of price series (spikes, mean reversion, multi-seasonality) are laid out cleanly here.

### The Iberian / Spanish market specifically

This is mostly grey literature, and reading it is genuinely part of the work — there's no textbook on OMIE.

- **OMIE market rules** ("Reglas de funcionamiento del mercado diario e intradiario"), on omie.es. Dry, but this is the actual algorithm your model is trying to predict.
- **REE, _Informe del Sistema Eléctrico Español_** — annual. Read three or four years apart (say 2018, 2021, 2024, 2025) and the structural transition of the Spanish system becomes vivid.
- **CNMC** market monitoring reports and the reports on the functioning of the wholesale market. The regulator's own diagnosis of what's driving prices.
- **RDL 10/2022** and CNMC/EC evaluations of the *mecanismo ibérico de ajuste* ("Iberian exception", Jun 2022 – Dec 2023). Read the legal instrument itself, and then someone's critique of it. This is the single largest structural artefact in any long Spanish price dataset.
- **MIBEL** regulatory council publications for the Spain–Portugal coupling and market splitting.
- **Post-blackout (28 Apr 2025) analysis**: ENTSO-E investigation reports, REE/Red Eléctrica statements, CNMC. A live topic with real operational consequences for 2025–26 price formation.
- **Ember, Montel, AleaSoft, OMIE monthly summaries** — the trade press. Not academic, but the fastest way to learn what practitioners consider salient right now (negative price hours, capture rates, evening ramp scarcity).

### EU market design and coupling

- **CACM Regulation (EU) 2015/1222**, and ENTSO-E's SDAC pages. The legal skeleton of the single day-ahead coupling.
- **Euphemia public description** (N-SIDE / NEMO Committee). The actual clearing algorithm — block orders, complex orders, PUN, and why the price is not simply "where two curves cross".
- **15-minute MTU transition** (live 1 Oct 2025). NEMO/EPEX/ENTSO-E documentation. Almost nothing academic exists on its price-formation effects yet, which is either a risk or an opportunity depending on temperament.

### Renewables economics — the merit-order effect and cannibalisation

*This is the literature your interpretability results will speak to. If you know it, your SHAP plots become arguments; if you don't, they're decoration.*

- **Sensfuß, Ragwitz & Genoese (2008)**, *Energy Policy* — the canonical merit-order effect paper.
- **Gelabert, Labandeira & Costa (2011)**, *Energy Economics* — ex-post analysis of renewables and cogeneration on Spanish prices. Directly your market.
- **Würzburg, Labandeira & Linares (2013)**, *Energy Economics* — review of merit-order effect estimates, Germany/Austria.
- **Hirth (2013)**, *"The market value of variable renewables"*, *Energy Economics* — why solar's own output depresses solar's revenue. The theoretical basis of cannibalisation.
- **López Prol, Steininger & Zilberman (2020)**, *Energy Economics* — cannibalisation in California. The clearest empirical treatment; Spain in 2026 looks like California in 2019 but faster.
- **Joskow (2011)**, *American Economic Review* — comparing costs of intermittent vs dispatchable generation. Short, sharp, foundational for why LCOE is misleading.
- **Negative prices**: search for recent work on drivers and frequency of negative prices in European markets. This is a fast-moving literature and Spain has become one of its most extreme cases.

⚔️ **Live disagreement:** how much of the observed price decline in high-RES hours is a *merit-order effect* (a real supply-curve shift) versus a *market-design artefact* (inflexible must-run, support schemes that make negative bidding rational, missing storage). Different framings give different answers about what your model has learned.

---

## PART 2 — Time series forecasting: foundations

*You have the ML background but possibly not the forecasting-specific background. The forecasting community has its own hard-won norms — especially around evaluation — that the ML community routinely violates.*

### Core

- **Hyndman & Athanasopoulos, _Forecasting: Principles and Practice_ (3rd ed.)** — free at otexts.com/fpp3. Start here. Chapters on evaluation, cross-validation for time series, and hierarchical forecasting are the most valuable for you.
- **Box, Jenkins, Reinsel & Ljung, _Time Series Analysis: Forecasting and Control_** — the classical reference. Skim unless you go the statistical route.
- **Hamilton, _Time Series Analysis_** (1994) — econometrics-grade, heavy. Reference only.

### Evaluation — the part most ML people get wrong

- **Diebold & Mariano (1995)**, *JBES* — comparing predictive accuracy. Plus **Diebold (2015)**, *"Comparing predictive accuracy, twenty years later"*, which is Diebold telling people they've been misusing his own test.
- **Giacomini & White (2006)**, *Econometrica* — conditional predictive ability. Handles nested models and recalibration properly.
- **Hyndman & Koehler (2006)**, *IJF*, *"Another look at measures of forecast accuracy"* — why MAPE is broken and what to use instead. Extremely relevant given Spanish prices now sit at or below zero for hundreds of hours a year.
- **Gneiting & Raftery (2007)**, *JASA* — strictly proper scoring rules. The theory of what makes an evaluation metric honest.
- **Gneiting & Katzfuss (2014)**, *Annual Review of Statistics* — probabilistic forecasting overview.

⚔️ **Live disagreement — cross-validation on time series:** the received wisdom is "never use k-fold, always use rolling origin." But **Bergmeir & Benítez (2012)**, *Information Sciences*, and **Bergmeir, Hyndman & Koo (2018)**, *CSDA*, argue that under certain conditions standard CV is valid and more efficient for purely autoregressive setups. Against this, **Tashman (2000)**, *IJF*, on out-of-sample testing, and the general practitioner consensus. Worth forming your own view, because it directly determines how much data you burn on validation.

### The competitions — evidence, not opinion

- **Makridakis, Spiliotis & Assimakopoulos (2020)**, M4 results, *IJF*. And the M4 commentary papers, which are a genuine argument.
- **Makridakis et al. (2022)**, M5 results, *IJF*. Notable: gradient boosting dominated; the winning solutions were heavily feature-engineered.
- **Makridakis et al. (2018)**, *"Statistical and machine learning forecasting methods: Concerns and ways forward"*, PLoS ONE — the paper that provoked the ML forecasting community. Read the rebuttals too.

⚔️ This is the deepest live fault line in the field: **does modern ML actually beat well-tuned simple methods on forecasting problems, or does it only appear to because of weak baselines?** M4 said "hybrid won, pure ML lost." M5 said "gradient boosting won decisively." Neural forecasting papers say "our architecture is SOTA." Read the primary evidence and decide.

---

## PART 3 — Electricity price forecasting (EPF) as a field

*This is your literature. It's small enough to actually map in a few weeks.*

### The spine

- **Weron (2014)**, *"Electricity price forecasting: A review of the state-of-the-art with a look into the future"*, *International Journal of Forecasting* 30(4):1030–1081. The field's founding review. Long; read the taxonomy sections carefully.
- **Lago, Marcjasz, De Schutter & Weron (2021)**, *Applied Energy* 293:116983 — the open-access benchmark paper, plus the `epftoolbox` library on GitHub. Its central complaint (weak baselines, no significance testing, private data) is the field's self-diagnosis. Note its benchmark covers EPEX-BE/FR/DE, NordPool and PJM — not Spain.
- **Jędrzejewski, Lago, Marcjasz & Weron (2022)**, *"Electricity price forecasting: The dawn of machine learning"*, *IEEE Power and Energy Magazine* 20(3):24–31. Short, accessible state-of-play.
- **Ziel & Weron (2018)**, *Energy Economics* — univariate vs multivariate modelling frameworks for day-ahead prices. Directly relevant to the "24 separate models vs one multi-output model" decision.

### Recent reviews (2025–2026) — use to map the current frontier

- **Jasiński (2025)**, *Energies* 18(24):6422 — review of deep learning in EPF, covering publications from 2023 onward, with error-metric summaries by market.
- ***Energies* 18(12):3097 (2025)** — review of point forecasting across day-ahead, intraday and balancing markets.
- **arXiv:2602.10071 (2026)** — deep learning for EPF across day-ahead, intraday and balancing markets. Recent enough to catch what the 2021 benchmark missed.
- **Mahtout & Ziel (2026)**, arXiv:2601.02856 — bridging linear models, neural networks and online learning. Ziel is one of the sharpest methodologists in this field; anything of his is worth reading.

### Methodological threads worth knowing exist

- **Variance-stabilising transformations**: Uniejewski, Weron & Ziel (2018), *IEEE Trans. Power Systems*. Which transform to apply to a spiky, occasionally-negative price series. Directly consequential now that log is unusable in Spain.
- **Calibration-window averaging**: Hubicka, Marcjasz & Weron (2019), *IEEE Trans. Sustainable Energy*. Cheap accuracy from averaging models trained on different window lengths.
- **Forecast combination generally**: Bates & Granger (1969) is the origin; the modern literature is large. "Averaging beats selecting" is one of forecasting's most robust empirical findings — worth deciding whether you believe it.
- **Using bid/order data**: Ciarreta, Martinez & Nasirov (2022), *IJF*, *"Forecasting electricity prices using bid data"* — Spanish authors, and the aggregated supply/demand curves OMIE publishes are an underexploited data source.
- **Fundamental/structural models**: an alternative paradigm where you *simulate* the merit order rather than regress on it. See work on hybrid fundamental-stochastic models (e.g. arXiv:2304.09336). Relevant because a structural model is interpretable *by construction* — a real alternative to black-box-plus-SHAP that you should consider before dismissing.
- **Effects of fuel and CO₂ prices**: Madadkhani & Ikonnikova (2024), *Energy Economics* 129:107241 — ML approach to quantifying high fuel and CO₂ price effects.

### Probabilistic EPF

- **Nowotarski & Weron (2018)**, *Renewable and Sustainable Energy Reviews* 81:1548–1568 — review of probabilistic EPF. The natural entry point.
- **Nowotarski & Weron (2015)** on Quantile Regression Averaging (QRA) — simple, strong, widely used.
- **Marcjasz, Narajewski, Weron & Ziel (2023)**, distributional neural networks for EPF, *Energy Economics* — modelling the whole predictive distribution rather than the mean.
- **Hirsch & Ziel (2024–2025)**, online distributional regression (arXiv:2407.08750, arXiv:2504.02518) — the current frontier of high-dimensional probabilistic EPF.
- **Brusaferri et al.** — probabilistic EPF with Bayesian/deep approaches.

⚔️ **Worth deciding early:** point forecast or distributional forecast? Spanish prices in 2026 have a bimodal, heavy-tailed, sometimes-negative distribution. There is a defensible argument that a conditional mean is close to meaningless for such a series, and an equally defensible argument that adding distributional machinery buries the interpretability story under complexity.

---

## PART 4 — The modelling toolbox and its arguments

*I've deliberately laid this out so you can adjudicate the LSTM question — and every equivalent question — from primary sources.*

### 4a. Regularised linear models

- **Hastie, Tibshirani & Friedman, _The Elements of Statistical Learning_** (free PDF) — chapters 3 and 7 especially. And **James et al., _An Introduction to Statistical Learning_** (free) if you want it gentler.
- **Tibshirani (1996)** — the original LASSO paper.
- The **LEAR** model (LASSO-estimated autoregressive) as specified in Lago et al. (2021). Reading the actual feature construction — ~247 features per hourly model — is instructive regardless of what you eventually use.
- **Hastie, Tibshirani & Wainwright, _Statistical Learning with Sparsity_** (free) if you want the theory of why L1 selects.

### 4b. Gradient boosting and the tabular-data question

- **Friedman (2001)**, *Annals of Statistics* — gradient boosting machines, the original.
- **Chen & Guestrin (2016)** XGBoost, KDD; **Ke et al. (2017)** LightGBM, NeurIPS; **Prokhorenkova et al. (2018)** CatBoost, NeurIPS. Read at least one properly — the engineering details (histogram binning, leaf-wise growth, ordered target statistics) matter for how you interpret them later.
- **Grinsztajn, Oyallon & Varoquaux (2022)**, NeurIPS, *"Why do tree-based models still outperform deep learning on typical tabular data?"*
- **Shwartz-Ziv & Armon (2022)**, *Information Fusion*, *"Tabular data: Deep learning is not all you need."*
- **Against those**: the deep-tabular line — TabNet (Arik & Pfister, 2021), FT-Transformer (Gorishniy et al., 2021), TabPFN (Hollmann et al., *Nature* 2025). TabPFN in particular is a serious challenge to the "trees always win" position on small tabular datasets.

⚔️ Genuinely unresolved. The honest summary is that it depends on dataset size, feature count, and how much tuning each side got — and most papers on both sides under-tune the opponent.

### 4c. Neural sequence models — the LSTM question and everything after it

**The recurrent lineage:**
- **Hochreiter & Schmidhuber (1997)** — the original LSTM. Worth reading once for what problem it was actually designed to solve.
- **Cho et al. (2014)** — GRU.
- **Greff et al. (2017)**, *"LSTM: A search space odyssey"*, IEEE TNNLS — systematic ablation of LSTM variants. Sobering.
- **Sutskever, Vinyals & Le (2014)** — seq2seq.
- **Salinas et al. (2020)**, *IJF* — DeepAR. Probabilistic, autoregressive RNN; still a strong industrial baseline.

**Convolutional alternatives:**
- **Bai, Kolter & Koltun (2018)**, *"An empirical evaluation of generic convolutional and recurrent networks for sequence modeling"* — the TCN paper, which argues convolutions beat RNNs on most sequence tasks. Directly relevant to whether you should use an LSTM at all.
- **van den Oord et al. (2016)** — WaveNet, the dilated-causal-convolution ancestor.

**Feedforward / basis-expansion:**
- **Oreshkin et al. (2020)**, ICLR — N-BEATS. Pure feedforward, beat statistical methods on M4, and has an *interpretable* variant with trend/seasonality basis blocks — relevant to your theme.
- **Challu et al. (2023)**, AAAI — N-HiTS.

**Transformers:**
- **Vaswani et al. (2017)** — attention. You've likely read it.
- **Lim, Arık, Loeff & Pfister (2021)**, *IJF*, *"Temporal Fusion Transformer"* — the architecture with built-in variable selection networks and interpretable attention. Sits precisely at the intersection of your two objectives.
- **Zhou et al. (2021)** Informer; **Wu et al. (2021)** Autoformer; **Zhou et al. (2022)** FEDformer — the long-sequence transformer wave.
- **⚔️ Zeng et al. (2023)**, AAAI, *"Are Transformers Effective for Time Series Forecasting?"* — shows a one-layer linear model (DLinear) matching or beating them all. One of the most useful papers in the field for developing scepticism.
- **Nie et al. (2023)**, ICLR — PatchTST, the response to that critique.
- **Liu et al. (2024)**, ICLR — iTransformer, inverting the attention axis.
- **Chen et al. (2023)** — TSMixer, MLP-only, competitive.

**Foundation models (2024–2026):**
- **Ansari et al. (2024)** — Chronos (Amazon), tokenised time series with a T5 backbone; Chronos-2 adds covariates and multivariate support.
- **Das et al. (2024)** — TimesFM (Google).
- **Woo et al. (2024)** — Moirai (Salesforce); Moirai-2.0 (2025).
- **Rasul et al. (2024)** — Lag-Llama.
- **PriceFM** (arXiv:2508.04875) — a foundation model specifically for probabilistic electricity price forecasting. Very close to your problem; worth a careful look.
- **GIFT-Eval** and **LOTSA** — the benchmark/dataset infrastructure. Read benchmark papers sceptically: zero-shot claims are sensitive to whether the evaluation series leaked into pretraining.

⚔️ **The question you actually asked.** The evidence you'd want to weigh: (1) the EPF-specific finding that a well-regularised LASSO is hard to beat; (2) the TCN paper's claim that convolutions dominate RNNs; (3) DLinear's claim that transformers were never justified; (4) M5's demonstration that boosted trees plus feature engineering wins tabular-flavoured problems; (5) against all of that, the steady stream of EPF papers reporting LSTM/CNN-LSTM/attention-LSTM improvements — many of which have weak baselines and no significance testing, but not all. Read a few LSTM-for-EPF papers *and* check whether they compare against a naive benchmark. That single check will teach you more about the field than any review.

### 4d. Hybrid and physics-informed

- **Physics-informed neural networks** (Raissi, Perdikaris & Karniadakis, 2019, *JCP*) — the general idea of embedding known constraints. In EPF this shows up as monotonicity constraints, merit-order structure, or bounded outputs.
- **Monotonic constraints** in LightGBM/XGBoost — a cheap, underused way to bake economics into a model (e.g. price must be non-decreasing in gas price, holding all else equal). Also an interpretability decision.
- **Structural/fundamental hybrid models** — see Part 3.

### 4e. Hyperparameter optimisation

- **Bergstra & Bengio (2012)**, JMLR — random search beats grid search.
- **Bergstra et al. (2011)** — Tree-structured Parzen Estimator, which is what `hyperopt` and Optuna use and what Lago et al. use for the DNN benchmark.
- **Akiba et al. (2019)** — Optuna.
- Practical: read about nested cross-validation and why tuning on your test set silently inflates everything.

---

## PART 5 — Interpretability / XAI

*Read the foundations and the critiques together. The critiques are more interesting.*

### Books

- **Molnar, _Interpretable Machine Learning_ (2nd ed., 2022)** — free at christophm.github.io/interpretable-ml-book. The single best entry point. Covers PDP, ICE, ALE, permutation importance, LIME, SHAP, counterfactuals, surrogate models, and — importantly — the limitations of each.
- **Molnar, _Interpreting Machine Learning Models With SHAP_ (2023)** — a whole short book on SHAP alone, including the correlation problems.
- **Biecek & Burzykowski, _Explanatory Model Analysis_** (CRC, free online) — more structured, with a nice framework for combining local and global explanations.
- **Barredo Arrieta et al. (2020)**, *Information Fusion* — the big XAI taxonomy review. Useful for vocabulary.

### Method papers

- **Shapley (1953)** — the original cooperative game theory. Short. Reading it makes clear what the axioms actually guarantee, and what they don't.
- **Lundberg & Lee (2017)**, NeurIPS — SHAP.
- **Lundberg et al. (2020)**, *Nature Machine Intelligence* 2:56–67 — TreeSHAP, exact and fast for tree ensembles.
- **Ribeiro, Singh & Guestrin (2016)**, KDD — LIME. And **Ribeiro et al. (2018)**, AAAI — Anchors.
- **Friedman (2001)** — partial dependence plots (in the GBM paper). **Goldstein et al. (2015)** — ICE plots.
- **Apley & Zhu (2020)**, *JRSS-B* — Accumulated Local Effects. The standard answer to PDP's failure under correlated features.
- **Breiman (2001)** — permutation importance (in the Random Forests paper). **Fisher, Rudin & Dominici (2019)**, JMLR — model reliance, the rigorous version.
- **Sundararajan, Taly & Yan (2017)** — Integrated Gradients, for differentiable models.
- **Friedman & Popescu (2008)** — H-statistic for interaction strength.
- **Wachter, Mittelstadt & Russell (2017)** — counterfactual explanations.

### ⚔️ The critiques — read these before you commit to SHAP

- **Rudin (2019)**, *Nature Machine Intelligence* 1:206–215, *"Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead."* The strongest argument that your entire methodology is the wrong approach. You should have an answer to it.
- **Kumar et al. (2020)**, ICML, *"Problems with Shapley-value-based explanations as feature importance measures."*
- **Sundararajan & Najmi (2020)**, ICML, *"The many Shapley values for model explanation"* — there is no single SHAP; the choices you make change the answer.
- **Chen, Janizek, Lundberg & Lee (2020)**, *"True to the Model or True to the Data?"* — interventional vs observational Shapley values. This is *the* paper for your correlated-features problem.
- **Slack et al. (2020)**, AIES, *"Fooling LIME and SHAP"* — adversarial attacks on post-hoc explanations.
- **Adebayo et al. (2018)**, NeurIPS, *"Sanity checks for saliency maps"* — some explanation methods produce the same output for a trained and a randomised model.
- **Doshi-Velez & Kim (2017)**, *"Towards a rigorous science of interpretable machine learning"* — the field still doesn't agree on what "interpretable" means or how to measure it.
- **Lipton (2018)**, *"The mythos of model interpretability"*, ACM Queue — sharp philosophical dissection of the word itself.

### Inherently interpretable alternatives (Rudin's answer)

- **Generalized Additive Models**: Hastie & Tibshirani. Then **Lou, Caruana & Gehrke (2012, 2013)** on GA²M / intelligible models.
- **Explainable Boosting Machines** and **InterpretML** (Nori et al., 2019) — GA²M with boosting; often competitive with black boxes on tabular data, and interpretable by construction.
- **Neural Additive Models** (Agarwal et al., 2021, NeurIPS) and **NODE-GAM**.
- **RuleFit** (Friedman & Popescu, 2008); **Optimal decision trees / sparse scoring systems** (Rudin's group).
- **Symbolic regression** (PySR, Cranmer et al.) — fit an explicit equation. Given that the underlying process has real closed-form economics (marginal cost of a CCGT), this is a more interesting option here than in most domains.

### XAI for time series specifically (harder, less mature)

- Surveys on XAI for time series — the core difficulty is that features are temporally dependent and permuting them breaks the data manifold, so most tabular XAI assumptions fail.
- **TFT** (Part 4c) as the "interpretable by architecture" route for sequences.
- **TimeSHAP**, **Dynamask**, and related temporal-attribution methods.
- **arXiv:2512.20514 (2025)** — sampling-free SHAP for transformers on time series.

### XAI in energy and power systems

- **Machlev et al. (2022)**, *Energy and AI* 9:100169 — XAI techniques for energy and power systems: review, challenges and opportunities. The standard reference.
- **Baur et al. (2024)**, *Energy and AI* 16:100358 — explainability and interpretability in electric load forecasting. Closest methodological neighbour to your project.
- ***MAKE* (2026), doi:10.3390/make8070188** — systematic PRISMA review of XAI in solar, wind and load forecasting, 154 articles, 2019–2026. Good for seeing what has and hasn't been tried.
- **Chen, Xiao, Guo & Yan (2023)**, *Advances in Applied Energy* 9:100123 — interpretable ML for building energy management. Different application, transferable methodology.

---

## PART 6 — Causality

*Include this because the gap between "the model uses feature X" and "X drives the market price" is a causal gap, and the XAI literature is mostly silent on it. Even a light reading here will sharpen how you write your conclusions — or convince you that a causal framing is the better project.*

- **Pearl & Mackenzie, _The Book of Why_** (2018) — popular, fast, genuinely reorients how you think. Good first contact.
- **Peters, Janzing & Schölkopf, _Elements of Causal Inference_** (MIT Press, free PDF) — short, rigorous, ML-oriented.
- **Hernán & Robins, _Causal Inference: What If_** (free) — the epidemiology tradition, very clear on confounding and identification.
- **Angrist & Pischke, _Mostly Harmless Econometrics_** — the applied-economics toolkit (IV, diff-in-diff, RDD). The Iberian gas cap is a near-textbook natural experiment, if that appeals.
- **Chernozhukov et al. (2018)**, *Econometrics Journal* — double/debiased machine learning.
- **Wager & Athey (2018)**, *JASA* — causal forests, heterogeneous treatment effects.
- **Athey & Imbens (2019)**, *Annual Review of Economics* — ML methods economists should know.
- **Schölkopf et al. (2021)**, *"Towards causal representation learning"* — where ML and causality are converging.

⚔️ There's an open argument about whether Shapley-value attributions can ever be given causal meaning. See **Heskes et al. (2020)** on causal Shapley values, and **Janzing, Minorics & Blöbaum (2020)** on the feature-relevance-as-intervention view — they disagree about the right formalism.

---

## PART 7 — Uncertainty quantification

- **Koenker & Bassett (1978)**, *Econometrica* — quantile regression. **Koenker, _Quantile Regression_** (2005) for the book treatment.
- **Meinshausen (2006)**, JMLR — quantile regression forests.
- **Angelopoulos & Bates (2021)**, *"A gentle introduction to conformal prediction and distribution-free uncertainty quantification"* — distribution-free prediction intervals with finite-sample guarantees, and there's specific work on conformal prediction for time series (adaptive conformal inference, Gibbs & Candès 2021).
- **Gal & Ghahramani (2016)** — MC dropout, if you go the Bayesian-neural route.
- **Pinball loss, CRPS, reliability diagrams** — covered in Gneiting & Raftery and the Nowotarski & Weron review.

---

## PART 8 — Data, tools, and documentation (reading of a different kind)

*Documentation is underrated as study material. Reading the ESIOS indicator catalogue teaches you more about the Spanish system than several papers.*

### Data sources
- **ESIOS (REE)** — `api.esios.ree.es`, free token. Read the full indicator catalogue at least once; it's an inventory of what the system operator considers measurable.
- **OMIE** — public file downloads (`marginalpdbc`, `curva_pbc`, etc.). The aggregated bid curves are the most interesting and least used.
- **ENTSO-E Transparency Platform** — free token, pan-European, includes published forecasts. The `entsoe-py` Python client is well-maintained.
- **MIBGAS** — Iberian gas indices, free historical download.
- **EU ETS carbon prices** — Ember, EEX, ICE.
- **Open-Meteo** — including the historical *forecast* archive (as opposed to reanalysis), which matters if you care about what was knowable at a given time.
- **ERA5 / Copernicus CDS** — reanalysis, excellent quality, but understand what reanalysis is before you use it as a predictor.

### Libraries worth knowing exist
- **`epftoolbox`** — LEAR, DNN, DM/GW tests, benchmark datasets.
- **`shap`**, **`interpret`** (InterpretML/EBM), **`alibi`**, **`dalex`**, **`PyALE`**, **`captum`** (for PyTorch attribution).
- **`darts`**, **`sktime`**, **`GluonTS`**, **`pytorch-forecasting`** (has TFT), **Nixtla** (`statsforecast`, `neuralforecast`, `mlforecast`) — the last is unusually fast and has strong baselines built in.
- **`optuna`**, **`mlflow`** or **`wandb`**, **`dvc`** for data versioning, **`hydra`** for config.
- **`pysr`** for symbolic regression.

---

## PART 9 — Research craft

*Cheap to read, disproportionate effect on the quality of the final document.*

- **Keshav, _"How to Read a Paper"_** — mentioned above.
- **Gelman & Loken (2013)**, *"The garden of forking paths"* — how honest researchers produce false results through undisclosed flexibility. Directly applicable to a project where you'll try many models.
- **Ioannidis (2005)**, *"Why most published research findings are false."*
- **Hullman & Gelman** on visualisation and uncertainty; **Tufte, _The Visual Display of Quantitative Information_** — your SHAP figures will carry the argument, so this matters more than usual.
- **Wilson et al., _"Good enough practices in scientific computing"_** (PLoS Comp Biol, 2017) — practical reproducibility.
- **The Turing Way** (free online handbook) — reproducible research practice.
- **Zinsser, _On Writing Well_** or Strunk & White, if English writing is part of the deliverable.

---

## PART 10 — A possible orientation over six months

*Offered as one plausible shape, not a plan. Adjust freely.*

| Phase | Rough focus | Reading centre of gravity |
|---|---|---|
| **Weeks 1–4** | Understand the market you're modelling | Part 1 (one book + Spanish grey literature), skim Part 3 spine |
| **Weeks 5–8** | Understand what forecasting demands | Part 2 (FPP3 + evaluation papers), Weron 2014, Lago 2021 |
| **Weeks 9–12** | Understand the model space and decide | Part 4 — read the debates, run small experiments |
| **Weeks 13–18** | Understand what explanation means | Part 5, especially the critiques; Part 6 lightly |
| **Weeks 19–24** | Produce, iterate, write | Part 9; revisit Parts 1 and 5 for the discussion chapter |

A useful discipline: keep a running document of **claims you've seen asserted with contradictory evidence**. By month three you'll have a list of live questions, and the shape of your thesis is likely to fall out of which ones you find most interesting.

---

## PART 11 — Open questions worth keeping a notebook on

These are unresolved in the literature as far as I can tell. Any one of them could become the spine of a thesis; none of them is a recommendation.

1. Does a model's learned sensitivity to gas price recover the physical heat rate of the marginal plant? If so, in which hours, and has that changed over the last decade?
2. Do post-hoc attributions on an EPF model agree with the merit-order-effect estimates obtained by econometric methods on the same market and period? Where they disagree, which is wrong?
3. Is an inherently interpretable model (EBM, GAM, sparse LEAR, symbolic regression) actually worse at forecasting Spanish prices than a black box — and if the gap is small, what justifies the black box?
4. How did the Iberian gas cap change the *learned* relationship between gas and price, and can that shift be detected in attributions without being told the policy dates?
5. Does 15-minute resolution reveal price-formation structure that hourly averaging destroys?
6. Do zero-shot foundation models beat a tuned LEAR on OMIE, and does the answer change during structural breaks?
7. What drives *uncertainty* in Spanish prices, as opposed to the level — and are the drivers different?
8. Negative-price hours are a different economic regime (opportunity cost of curtailment, support-scheme incentives, must-run inflexibility). Should they be modelled as a separate regime, and what do attributions look like inside that regime?

---

*Nothing here obliges you to anything. If a section looks uninteresting after twenty minutes of skimming, that's information — the areas that pull you are the ones likely to survive six months of work.*
