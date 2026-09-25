# Cognitive-Neuroscience Task Battery for a Multi-Task AL-RNN

A battery built **strictly from cognitive-neuroscience paradigms** — every task traceable to an
animal-electrophysiology or human cognitive-psychology experiment (plus a small set of
theoretical-neuroscience dynamical probes). Target: **~80–100 encoded tasks** on one shared
supervised interface, ordered along a complexity axis from near-linear to attractor-dependent.

Lineage: **Yang et al. 2019** → **Khona, Chandra, Bhaskar & Fiete (Mod-Cog) 2022** →
**Driscoll, Shenoy & Sussillo 2024**, broadened across the cognitive-neuroscience canon.

> **Scope rule:** pure machine-learning RNN benchmarks (copy, sorting, adding problem, seq-MNIST,
> modular arithmetic, Lorenz) are **excluded** from the cognitive battery and live in the
> optional-controls appendix (§Z) only — they are not counted toward the 80–100.

---

## How to read this catalog

**name** — description · *experimental origin* · demand · fit · build-status

**Dynamical-demand** (the complexity ordinate): `°` plausibly near-linear · `★` requires
recurrence/memory · `★★` canonical attractor/limit-cycle/fixed-point-set probe.

**Interface fit:** `[FIT]` direct · `[FIT*]` minor encoding choice · `[OUT]` needs wider output.

**Build:** `[have]` implemented · `[mod]` via a modifier · `[new]` to build.

---

## A. Stimulus–response & inhibitory control
*Saccade/reach neurophysiology; Munoz & Everling 2004; Logan & Cowan 1984.*
1. **Go** · *Hikosaka & Wurtz* · ° · [FIT] [have/mod]
2. **Reaction-time Go** · *Yang19* · ° · [FIT] [mod]
3. **Delay Go (Pro)** — remember direction, respond after delay · *Yang19* · ★ · [FIT] [have]
4. **Anti(saccade)** — respond opposite stimulus · *Munoz & Everling 2004* · ° · [FIT] [have]
5. **RT Anti** · *Yang19* · ° · [FIT] [mod]
6. **Delay Anti** · *Yang19* · ★ · [FIT] [have]
7. **Go/No-Go** — respond or withhold · *Sakagami & Niki 1994* · ° · [FIT] [have]
8. **AntiReach / Reaching-1D** · *Georgopoulos* · ° · [FIT] [new]
9. **Stop-signal** — cancel a prepared response · *Logan & Cowan 1984* · ★ · [FIT] [new]
10. **Countermanding** — race-to-cancel saccade · *Hanes & Schall 1996* · ★ · [FIT*] [new]

## B. Perceptual decision-making / evidence integration
*Gold & Shadlen 2007; Mante 2013; Brunton & Brody 2013.*
11. **Perceptual DM (random dots)** · *Roitman & Shadlen 2002* · ★ · [FIT] [have]
12. **PDM with delayed response** · *Yang19* · ★ · [FIT] [mod]
13. **Poisson-clicks accumulation** · *Brunton, Botvinick & Brody 2013* · ★ · [FIT] [have]
14. **Multisensory integration** · *Raposo, Kaufman & Churchland 2014* · ★ · [FIT] [have]
15. **Context-dependent DM** — selective integration · *Mante, Sussillo et al. 2013* · ★★ · [FIT] [have≈]
16. **Single-context DM** · *Yang19* · ★ · [FIT] [have≈]
17. **Multisensory delay DM** · *Yang19* · ★ · [FIT] [mod]
18. **Tone detection in noise** · *de Lafuente & Romo 2005* · ° · [FIT] [have]
19. **Spatial-suppression motion** — size-dependent DM · *Tadin et al. 2003* · ★ · [FIT] [new]
20. **Vibrotactile detection (yes/no)** · *de Lafuente & Romo 2005* · ★ · [FIT] [new]
21. **Post-decision wager / confidence report** · *Kiani & Shadlen 2009* · ★ · [FIT*] [new]
22. **Face/category discrimination (morph)** · *Freedman et al. 2001* · ★ · [FIT] [new]

## C. Working memory / match-to-sample / parametric WM
*Funahashi 1989; Romo 1999; Compte 2000; Freedman & Assad 2006.*
23. **Delay match-to-sample (DMS)** · *Miller, Erickson & Desimone 1996* · ★ · [FIT] [have]
24. **Delay non-match-to-sample (DNMS)** · *Yang19* · ★ · [FIT] [have]
25. **Delay match-to-category (DMC)** · *Freedman & Assad 2006* · ★ · [FIT] [have≈]
26. **Delay non-match-to-category (DNMC)** · *Yang19* · ★ · [FIT] [have≈]
27. **DMS with distractor** — robust attractor memory · *Miller et al. 1996* · ★★ · [FIT] [mod]
28. **Dual delay match-to-sample** · *Yang19* · ★ · [FIT] [have]
29. **Parametric WM / delay comparison (f1 vs f2)** · *Romo, Brody, Hernández & Lemus 1999* · ★★ · [FIT] [have]
30. **Delay paired association** · *Sakai & Miyashita 1991* · ★ · [FIT] [new]
31. **Sternberg item recognition** (set-size param) · *Sternberg 1966* · ★★ · [FIT] [new]
32. **N-back (1/2/3-back)** · *Kirchner 1958; Owen et al. 2005* · ★★ · [FIT] [have]
33. **Oculomotor delayed response / spatial-WM ring** · *Funahashi 1989; Compte 2000* · ★★ · [FIT*] [new]
34. **Change detection (array)** · *Luck & Vogel 1997* · ★ · [FIT*] [new]
35. **Serial / order recall (span)** · *digit/spatial span* · ★ · [FIT] [mod]
36. **Spatial-WM bump maintenance** · *Compte et al. 2000* · ★★ · [FIT*] [new]

## D. Timing / interval estimation & production
*Jazayeri & Shadlen 2010; Paton & Buonomano 2018.*
37. **Ready-Set-Go (motor timing)** · *Jazayeri & Shadlen 2010* · ★★ · [FIT] [have≈]
38. **One-Two-Three-Go** · *Sohn, Narain et al. 2019* · ★★ · [FIT] [new]
39. **Interval discrimination** · *NeuroGym* · ★ · [FIT] [have]
40. **Interval reproduction** · *Jazayeri & Shadlen 2010* · ★★ · [FIT] [new]
41. **Temporal bisection** · *Church & Deluty 1977* · ★ · [FIT] [new]
42. **Duration estimation (pro/anti)** · *have* · ★ · [FIT] [have]
43. **Rhythmic production at cued tempo** · *motor timing* · ★★ · [FIT*] [new]

## E. Sequence memory / production (cognitive, not ML)
*Nissen & Bullemer 1987; Hikosaka et al. (sequence learning).*
44. **Delayed serial recall** — reproduce an ordered list after delay · *cognitive span* · ★ · [FIT] [mod]
45. **Serial reaction-time task (SRTT)** — implicit sequence learning · *Nissen & Bullemer 1987* · ★ · [FIT] [new]
46. **Sequence reproduction / motor sequence** · *Hikosaka et al. 1999* · ★ · [FIT] [mod]
47. **Self-ordered search** — visit each item once · *Petrides & Milner 1982* · ★★ · [FIT*] [new]

## F. Cognitive control / rule-based / context switching
*Miller & Cohen 2001; Stroop 1935; Eriksen 1974.*
48. **Task switching (cued rule swap)** · *Sakai 2008; Sohn et al. 2000* · ★ · [FIT] [new]
49. **Stroop (conflict)** · *Stroop 1935* · ★ · [FIT] [new]
50. **Flanker** · *Eriksen & Eriksen 1974* · ° · [FIT] [new]
51. **Simon** · *Simon 1969* · ° · [FIT] [new]
52. **AX-CPT (context maintenance)** · *Servan-Schreiber et al. 1996* · ★ · [FIT] [new]
53. **Wisconsin Card Sort / rule reversal** — latent-rule inference · *Berg 1948; Milner 1963* · ★★ · [FIT] [new]
54. **Hierarchical / nested-rule reasoning** · *Badre & D'Esposito 2009* · ★★ · [FIT*] [new]
55. **Cue-based gating / routing** · *Miller & Cohen 2001* · ★ · [FIT] [new]
56. **Probabilistic reversal learning (supervised-feedback form)** · *Cools et al. 2002* · ★★ · [FIT] [new]

## G. Canonical dynamical-systems probes (theoretical neuroscience)
*Sussillo & Barak 2013; Sussillo & Abbott 2009; Driscoll/Sussillo 2024.*
57. **N-bit flip-flop** — discrete fixed-point memory (bitcode probe) · *Sussillo & Barak 2013* · ★★ · [FIT] [have]
58. **Oscillator / sine generation (frequency-cued)** — limit cycle · *Sussillo & Abbott 2009* · ★★ · [FIT] [have]
59. **Context-switching oscillation** — switch frequency on cue · *Driscoll/Sussillo 2024* · ★★ · [FIT] [new]
60. **Neural integrator (line attractor)** — accumulate a scalar to memory · *Seung 1996; Aksay et al.* · ★★ · [FIT*] [new]

## H. Spatial / navigation (bridges to the T-maze work)
*Hafting 2005; Cueva & Wei 2018; Banino 2018; Kim et al. 2017.*
61. **Path integration / dead reckoning** — velocity → position · *Cueva & Wei 2018; Banino et al. 2018* · ★★ · [FIT*] [new]
62. **Head-direction integration** — angular velocity → heading (ring attractor) · *Kim, Rouault, Druckmann & Jayaraman 2017* · ★★ · [FIT*] [new]
63. **Spatial alternation / T-maze** · *have benchmark* · ★ · [FIT] [new]
64. **2D navigation-to-remembered-goal** · *spatial WM + action* · ★ · [OUT] [new]
65. **Mental rotation** — transform a held spatial code · *Shepard & Metzler 1971* · ★★ · [FIT*] [new]
66. **Visual search (serial/parallel)** · *Treisman & Gelade 1980* · ★ · [FIT*] [new]

## I. Probabilistic / Bayesian inference & associative learning
*Knowlton 1994; Wilson et al. 2010; Ernst & Banks 2002.*
67. **Weather prediction (probabilistic categorization)** · *Knowlton, Squire & Gluck 1994* · ★ · [FIT] [new]
68. **Probabilistic reasoning (evidence weighting)** · *Yang & Shadlen 2007* · ★ · [FIT] [new]
69. **Cue-reliability-weighted integration** · *Ernst & Banks 2002* · ★ · [FIT*] [new]
70. **Change-point / volatility detection** · *Wilson, Nassar & Gold 2010* · ★★ · [FIT] [new]
71. **Acquired-equivalence / transitive inference** · *Dusek & Eichenbaum 1997* · ★★ · [FIT*] [new]

---

## J. Modifier multipliers (the scaling engine — themselves cognitive manipulations)
*Khona, Chandra, Bhaskar & Fiete (Mod-Cog) 2022. Each is a classic experimental manipulation.*
- **`+delay`** — insert a memory delay before response (reactive → working memory). `★`
- **`+int`** — correct output drifts continuously through the delay (demands **rotation**). `★★`
- **`+seq`** — single response becomes a time-varying motor output. `★`
- composition **`+int×+seq`** — the largest count multiplier.

---

## Headcount: reaching 80–100 (cognitive only)
- **Cognitive base paradigms catalogued (A–I):** **71** distinct base tasks.
- **Implemented now (cognitive subset of the registry):** ~22 base + keystones.
- **Path to target:** ~45–55 cognitive base tasks **+** modifier variants
  (`+delay`/`+int`/`+seq` on the ~25 delay/WM/DM-eligible bases) → **80–100**, every one a
  recognizable cognitive paradigm (e.g. *delayed* context-DM, *integration* DMC, *sequence* anti).
- **Excluded from the count:** RL/value tasks (§K) and all ML benchmarks (§Z).

## Complexity axis (the scientific ordinate)
`°` reactive/near-linear → `★` recurrence & memory → `★★` attractor / limit-cycle / continuous-attractor.
The `★★` probes (flip-flop, oscillator, parametric WM, spatial-WM bump, path/head-direction
integration, line attractor, `+int` variants) are exactly the tasks predicted to break near-linear
solvability — the falsifiable core of the argument.

---

## K. Reinforcement / value-based — DEFERRED (not in the battery)
Bandits, reversal, Daw two-step, post-decision wager, economic choice (*Padoa-Schioppa & Assad
2006*), Iowa gambling. No fixed per-timestep target; integrate later via meta-learning or real RL.

## Z. Optional non-cognitive controls (NOT counted; fenced for contrast only)
Pure ML-RNN benchmarks, useful only as deliberate *outliers* (your prior finding treats copy/seq
as an outlier cluster): copy / repeat-copy / sorting / associative recall (*Graves NTM/DNC*),
adding & multiplication problems, sequential & permuted MNIST, modular arithmetic, Lorenz
reconstruction. Keep separate from the cognitive battery; use only if a non-cognitive contrast is
explicitly wanted.
