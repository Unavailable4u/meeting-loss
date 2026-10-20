# Brain vs. Artificial Neural Networks: Learning and Memory Compared

> **Status note (2026-10-21):** This is a frozen background document and the project's **idea bank**. Ideas 1 to 3 in Section 8 (surprise-gated learning rate, homeostatic update normalization, metaplastic per-parameter rates) are the designated backup plan (D-009). Return here when the meeting-loss work reaches a gate that calls for a new idea. See `docs/DOCUMENT_MAP.md`.

*A working reference for finding brain-inspired mechanisms that could improve training efficiency.*

## 0. How to use this document

- **Purpose:** to map how biological brains learn and remember onto how models learn, and to turn the differences into testable research ideas (Section 8).
- **Spirit:** analogies are a source of *inspiration*, not proof. A mechanism being biological does not make it useful for a model, and many brain-inspired ideas have failed to beat plain backpropagation with Adam. Every idea here must still pass the rigorous experiments from your proposal (tuned baselines, multiple seeds, equal compute).
- **Accuracy note:** the neuroscience is simplified, and numbers are approximate. Verify specifics against the primary sources in Section 11 before putting them in a paper. Neuroscience also has many open questions, so some statements below are "current best understanding," not settled fact.

---

## 1. First, a correction: "digital vs. analog" is not the main difference

You said models are digital and brains are analog. That is partly true, but it is *not* the most useful way to see the gap, for two reasons.

**The brain is a hybrid, not purely analog.**

- Neurons communicate with **spikes** (action potentials). A spike is all-or-none: it either happens or it does not. In that sense it is closer to a digital event.
- But *when* spikes happen is continuous in time, and the effect of each spike is **graded and analog**: synapses have varying strengths, dendrites sum inputs in complex ways, and chemical concentrations vary smoothly.
- So the brain mixes digital-like signaling (spikes) with analog-like computation (integration, graded synaptic strength, continuous time).

**Models are digital hardware, but they mostly simulate continuous math.**

- A neural network is a set of real-valued equations computed with floating-point numbers. The hardware is digital, but the *algorithm* is continuous-valued. Analog and neuromorphic chips exist, and the same algorithms could in principle run on them.

**The better distinction is hardware versus algorithm.**

- The *substrate* (wet chemistry vs. silicon) matters for energy and speed.
- The *algorithm* (how learning rules, structure, and memory work) is where transferable ideas live. Almost everything in this document is about the algorithm level, because that is what you can change in a paper about training.

The large differences that matter for your research are: the **learning rule**, **energy use**, **data efficiency**, **memory organization**, and **continual learning**.

---

## 2. Building blocks: how the parts map onto each other

| Biological | Artificial | Notes |
|---|---|---|
| Neuron | Unit / node | A real neuron is far more complex than a unit. A single neuron's dendrites can perform computations that may need a small network to imitate. |
| Synapse | Weight (parameter) | A synapse has multiple states and timescales (short-term vs. long-term changes). A weight is one number. |
| Firing rate / spikes | Activation value | Most models pass continuous numbers. Spiking neural networks (SNNs) imitate spikes but are harder to train. |
| Dendritic integration | Weighted sum + nonlinearity | Biological dendrites have local nonlinearities, which is much richer than one sum. |
| Neurotransmitters | Mostly none | Excitatory vs. inhibitory is roughly the sign of a weight. Inhibition also has dedicated cell types and roles. |
| Neuromodulators (dopamine, acetylcholine, serotonin, noradrenaline) | Roughly: learning-rate or gating signals (no standard equivalent) | They broadcast "pay attention," "this was surprising," or "this was rewarded" and change *how much* learning happens. |
| Cortical layers and areas | Network layers / modules | The cortex has recurrent loops and feedback connections everywhere, not only a feed-forward stack. |
| Hippocampus | Roughly: episodic memory, external memory / replay buffer | Fast one-shot learning of specific events. |
| Neocortex | Roughly: the main trained weights | Slow learning of general structure. |
| Sleep and replay | Roughly: experience replay, rehearsal | Reactivates recent memories to strengthen and integrate them. |
| Evolution | Outer loop: architecture design, initialization, meta-learning | Evolution pre-installs strong priors, so a newborn does not learn from scratch. |

### Scale and resource comparison (very approximate)

| Quantity | Human brain | Large AI model |
|---|---|---|
| Computing elements | ~86 billion neurons | Billions to trillions of parameters |
| Connections | ~10^14 to 10^15 synapses (estimates vary widely) | Same as parameters |
| Power | ~20 watts | Training a large model uses megawatts over weeks or months, on the order of gigawatt-hours for GPT-3-class models, and more today |
| Activity | Sparse: only a small fraction of neurons fire at a time, at low average rates | Dense in most architectures: nearly all units compute for every input (mixture-of-experts and sparse models are partial exceptions) |
| Speed per element | Slow (milliseconds) | Fast (nanoseconds) |
| Parallelism | Massive, asynchronous, event-driven | Massive, synchronous, clocked |

Comparing "number of synapses" with "number of parameters" is rough: a synapse is more complex than a weight, so this comparison understates the brain's per-element capability.

---

## 3. How learning works

### 3.1 How a model learns (recap)

1. **Forward pass:** input flows through layers and produces a prediction.
2. **Loss:** one number measures how wrong the prediction was.
3. **Backward pass (backpropagation):** the gradient of the loss with respect to *every* weight is computed using the chain rule, sending an error signal backward through the network.
4. **Update:** each weight moves slightly against its gradient: `w ← w − η · ∂L/∂w` (with Adam and others adding momentum and per-parameter scaling).
5. Repeat over a huge dataset.

Key properties:

- **Global error signal:** a single loss drives every weight change.
- **Precise credit assignment:** backprop tells each weight exactly how much it contributed to the error.
- **Needs stored activations:** the backward pass requires the values from the forward pass, which costs memory.
- **Needs symmetric weights:** the backward pass uses the transpose of the forward weights. This is the **weight transport problem**.
- **Separate phases:** forward, then backward, then update. Learning and inference are different modes.
- **Frozen after training:** once deployed, the weights usually stop changing.

### 3.2 How the brain learns (current understanding)

The brain changes the strength of synapses, a process called **synaptic plasticity**. No single rule explains everything. Several mechanisms work together, at different timescales.

**Hebbian learning** ("cells that fire together wire together," Hebb 1949)

- If a neuron repeatedly helps cause another to fire, the connection strengthens.
- Simple form: `Δw_ij = η · x_i · y_j` (pre-synaptic activity × post-synaptic activity).
- Pure Hebb is unstable (weights grow without limit), so variants add a stabilizer. **Oja's rule:** `Δw = η · y · (x − y · w)`.

**Long-term potentiation and depression (LTP / LTD)**

- Synapses can strengthen (LTP) or weaken (LTD) for hours to years. This is thought to be a cellular basis of memory.

**Spike-timing-dependent plasticity (STDP)**

- The *order* of spikes matters. If the pre-synaptic neuron fires shortly *before* the post-synaptic one (it plausibly caused it), the synapse strengthens. If it fires after, the synapse weakens.
- `Δw = A₊ · exp(−Δt/τ₊)` if pre fires before post, and `Δw = −A₋ · exp(Δt/τ₋)` otherwise (with Δt the time difference).

**BCM rule and metaplasticity** (Bienenstock, Cooper, Munro 1982)

- The threshold between strengthening and weakening *slides* with the neuron's recent activity: `Δw = η · x · y · (y − θ)`, where `θ` tracks recent average activity.
- This is **metaplasticity**: plasticity itself being plastic. It stabilizes learning.

**Homeostatic plasticity / synaptic scaling**

- Neurons keep their overall activity near a target by scaling all their synapses up or down: roughly `w ← w · (r_target / r_actual)`. This prevents runaway excitation or silence while preserving *relative* weights.

**Neuromodulated (three-factor) learning**

- Plasticity often depends on a third signal: `Δw = η · e_ij(t) · M(t)`, where `e_ij` is an **eligibility trace** (a decaying record that "this synapse was recently active") and `M` is a broadcast modulatory signal such as dopamine.
- Dopamine neurons appear to signal **reward prediction error**, the same quantity as the temporal-difference error in reinforcement learning: `δ = r + γ·V(s′) − V(s)` (Schultz, Dayan, Montague 1997).
- This solves delayed credit assignment: the trace remembers which synapses were involved until the reward arrives.

**Predictive coding and error-driven local learning**

- A leading theory is that cortex continually predicts its own input, and neurons carry *prediction errors* (`ε = actual − predicted`). Learning reduces these errors using **local** information.
- Some predictive-coding networks have been shown to approximate backpropagation under certain conditions (Whittington and Bogacz 2017). Whether the brain does something like this is still debated.

**Structural plasticity**

- The brain grows and prunes synapses. Infants overproduce connections, and pruning refines the network during development. Wiring itself, not only strengths, is learned.

**Learning is also guided by:** attention, curiosity and novelty (we study what we do not yet understand), sleep, emotion (important events are stored more strongly), and active exploration (we choose what to experience).

### 3.3 Side by side

| Aspect | Model (backprop + optimizer) | Brain |
|---|---|---|
| Error signal | One global loss, propagated precisely to all weights | Mostly local signals, plus broadcast modulators (dopamine etc.). Exact mechanism for deep credit assignment is unresolved |
| Weight updates | Computed from exact gradients | Local rules (Hebbian, STDP) shaped by modulatory signals |
| Needs symmetric backward weights | Yes | No known mechanism for exact transport |
| Learning vs. inference | Separate phases | Continuous: learning happens while operating |
| Timing | Discrete steps | Continuous time, spike timing matters |
| Data per update | Mini-batches (many examples averaged) | One experience at a time, in a stream |
| Supervision | Mostly explicit labels or next-token targets | Mostly self-supervised prediction plus sparse reward and social feedback |
| Learning rate | Usually a global schedule | Modulated by surprise, attention, emotion, and development stage |
| Activity | Dense | Sparse, event-driven |
| Stabilization | Normalization layers, weight decay, clipping | Homeostasis, inhibition, metaplasticity |
| Structure | Fixed architecture (usually) | Wiring grown and pruned |
| After training | Weights frozen | Keeps learning for life |

---

## 4. How memory works

### 4.1 Brain memory systems (simplified)

- **Sensory memory:** a very brief trace of incoming senses (fractions of a second to seconds).
- **Working memory:** the small, active "workspace." Capacity is limited to roughly a handful of items (often estimated around 4 chunks). It is thought to be held through sustained neural activity and short-term synaptic changes.
- **Long-term memory:** stable, with several kinds:
  - **Episodic:** specific events ("what I ate yesterday"), strongly tied to the hippocampus.
  - **Semantic:** general facts and concepts, distributed across cortex.
  - **Procedural:** skills and habits, involving basal ganglia and cerebellum.
- **Consolidation:** new memories are first held in the hippocampus, then gradually integrated into cortex, helped by **replay** during sleep. This is the **Complementary Learning Systems** idea (McClelland, McNaughton, O'Reilly 1995): a fast learner for specifics plus a slow learner for general structure. The slow learner needs interleaved replay to avoid overwriting old knowledge.
- **Retrieval is reconstructive:** recalling a memory partly rebuilds it, and **reconsolidation** can modify it each time it is recalled. Memories are not exact recordings.
- **Content-addressable / associative:** a partial cue (a smell, a few words) can recover the whole memory. This is **pattern completion**, modeled by Hopfield networks (1982).
- **Forgetting is active and useful:** the brain discards or de-emphasizes details, which helps generalization.

### 4.2 Model memory

- **Parameters (weights):** long-term, slowly learned, *distributed* storage. There is no separate "episodic" store. Facts are smeared across billions of numbers, and individual memories cannot be located or edited easily.
- **Activations / context window:** act like working memory. A transformer's context (and its KV cache) holds the current "conversation," but disappears when the session ends.
- **External memory:** retrieval-augmented generation (RAG), vector databases, or files give an add-on episodic-like store, bolted on rather than learned into the network.
- **No consolidation after deployment:** what happens in a conversation does not update the weights. Memory across sessions needs external tools.
- **Catastrophic forgetting:** training on new data can overwrite old knowledge. The brain largely avoids this, through mechanisms like complementary learning systems and metaplasticity.
- **Attention as associative memory:** a mathematical link exists between **modern Hopfield networks** and transformer attention (Ramsauer et al. 2020). Attention can be seen as a content-addressable lookup, so the analogy to associative memory is more than loose.

### 4.3 Memory comparison table

| Aspect | Model | Brain |
|---|---|---|
| Long-term store | Weights, distributed, hard to edit | Synapses plus structure, also distributed, but with specialized systems |
| Fast, one-shot storage | Not built in (needs RAG or fine-tuning) | Hippocampus |
| Working memory | Context window / activations, large and precise | Small capacity, limited to a few items |
| Consolidation | None after training | Replay during sleep, gradual transfer |
| Recall | Exact lookup in context, approximate recall of weights | Reconstructive, cue-based, changeable |
| Forgetting | Accidental (catastrophic) | Regulated, adaptive |
| Capacity limits | Fixed by size | Limited but efficient, with compression and abstraction |

---

## 5. Efficiency and generalization

**Energy.** The brain runs on about 20 W. Training large models uses many orders of magnitude more energy. Part of this is hardware (digital chips move a lot of data), and part is algorithmic (dense computation, repeated passes over data). The algorithmic part is what you could target.

**Data efficiency.** A child learns a language from a small fraction of the text that large language models consume. Children hear roughly millions to tens of millions of words per year, while models train on trillions of tokens. The brain also learns concepts from a few examples. Reasons likely include:

- **Innate priors:** evolution built in structure (visual processing, a readiness for language, intuitive physics) that gives learning a head start. In ML terms, evolution acts as a very long meta-learning loop that sets architecture and initialization.
- **Multimodal, embodied experience:** the child sees, hears, touches, and acts, and the signals constrain each other.
- **Active learning:** children choose what to explore, often what they find surprising or just-learnable.
- **Curriculum and development:** learning proceeds from simple to complex, and low-resolution vision early in life may help.
- **Strong self-supervision:** constant prediction of what will happen next.
- **Sleep and replay:** repeated use of each experience.

**Continual learning.** Humans learn new things without forgetting most old ones. Models struggle with this.

**Robustness.** Brains handle noise, partial damage, and distribution shift gracefully, though they also have systematic biases and illusions.

### Where models are better

For balance, models outperform brains in several ways, and these are reasons not to copy biology blindly:

- **Exact precision and copying:** weights can be duplicated perfectly and shared.
- **Speed and scale:** huge datasets and parallel hardware.
- **Exact credit assignment:** backprop is a highly effective way to assign credit, possibly better than what biology does.
- **No fatigue, no biological constraints** (slow signal speed, limited energy supply, cell maintenance).

Biology solved *its* constraints (slow, noisy, energy-limited, no backward pass), and some of its tricks may not help on a GPU.

---

## 6. The biggest gaps (the "opportunity map")

| Gap | Brain does | Model does | Opportunity |
|---|---|---|---|
| Learning rate control | Modulated by surprise, attention, development | Mostly a fixed schedule | Surprise- or signal-driven adaptive learning rates |
| Stability while learning | Homeostasis, metaplasticity | Normalization layers, clipping, weight decay | New stabilizing update rules |
| Continual learning | Largely avoids catastrophic forgetting | Forgets | Importance-aware plasticity, replay |
| Fast + slow learning | Hippocampus + neocortex | One learning speed | Fast/slow weight schemes |
| Sparsity | Sparse, event-driven | Dense | Lower FLOPs through sparse activation or training |
| Credit assignment cost | Local rules (cheap memory) | Backprop (stores activations) | Local or approximate gradients |
| Data efficiency | Priors, curriculum, active selection | Brute-force data | Better data ordering and selection, meta-learned initialization |
| Structure learning | Growth and pruning | Fixed architecture | Dynamic sparse training |

---

## 7. What a "formula" inspired by biology looks like

Most brain-inspired learning rules share one shape:

```
Δw_ij = (learning rate) × (local signal)  ×  (modulating signal)  ×  (stabilizing term)
```

Compare with standard training:

```
Δw_ij = −η × ∂L/∂w_ij   (with momentum and per-parameter scaling in Adam)
```

Your research freedom is in what goes into each factor:

- **Local signal:** pre/post activity, a gradient, an eligibility trace
- **Modulating signal:** loss change, surprise, reward, layer-wise signal
- **Stabilizing term:** normalization, a sliding threshold, an importance penalty

---

## 8. Idea bank: mechanisms to test

Each idea lists the biological inspiration, a rough machine-learning translation, related existing work (so you check novelty), and what to measure. These are **starting hypotheses**, not novel claims. Most have some prior work, so a literature check is the first step.

### Idea 1: Surprise-gated learning rate (optimizer)

- **Inspiration:** dopamine and other neuromodulators scale learning by prediction error and surprise.
- **Translation:** scale the step size by how surprising the current loss or gradient is relative to its recent history, e.g. `η_t = η₀ · g(s_t)` with `s_t = (L_t − L̄_t) / σ_L`, where `L̄` and `σ_L` are running statistics.
- **Related:** adaptive learning-rate methods, loss-aware schedules, prioritized replay in RL.
- **Measure:** steps to target loss vs. tuned AdamW with a standard schedule.

### Idea 2: Homeostatic update normalization (optimizer / architecture)

- **Inspiration:** synaptic scaling keeps activity within a healthy range.
- **Translation:** after each update, rescale a layer's weights, activations, or update size toward a target value, e.g. `w ← w · (target_norm / current_norm)` or normalize the update magnitude per layer.
- **Related:** LayerNorm/BatchNorm, weight normalization, LARS/LAMB, Muon-style normalized updates.
- **Measure:** tolerance to learning rate (wider good range), speed to target, stability.

### Idea 3: Metaplastic, importance-aware plasticity (optimizer; continual learning)

- **Inspiration:** BCM and synaptic cascade models: synapses that matter become harder to change.
- **Translation:** per-parameter learning rate decreases with accumulated importance, `η_i = η₀ / (1 + α · Ω_i)`, where `Ω_i` tracks how much parameter `i` has mattered (for instance the running average of squared gradients, which relates to Adam's second-moment estimate and the Fisher information).
- **Related:** Elastic Weight Consolidation, Synaptic Intelligence, Benna-Fusi cascade model.
- **Measure:** forgetting on sequential tasks, plus training speed in the normal setting.

### Idea 4: Fast and slow weights (optimizer)

- **Inspiration:** hippocampus (fast) plus neocortex (slow).
- **Translation:** maintain fast weights updated every step and slow weights that move toward them periodically: `θ_slow ← θ_slow + α · (θ_fast − θ_slow)` every `k` steps.
- **Related:** the Lookahead optimizer, fast weights (Hinton and Plaut 1987; Ba et al. 2016), weight averaging.
- **Measure:** stability, final quality, and sensitivity to learning rate. Because Lookahead exists, a contribution would need a clearly different mechanism.

### Idea 5: Sparse activation and winner-take-all (architecture)

- **Inspiration:** sparse firing, lateral inhibition.
- **Translation:** keep only the top-k activations per layer (k-winner-take-all), or route each input to a few experts.
- **Related:** top-k activations, mixture-of-experts, sparse transformers.
- **Measure:** quality at equal FLOPs, or FLOPs saved at equal quality.

### Idea 6: Local learning and predictive coding (gradient computation)

- **Inspiration:** local plasticity driven by prediction errors, no global backward pass.
- **Translation:** each layer minimizes its own local error using only nearby information, so activations need not all be stored.
- **Related:** predictive coding networks, Forward-Forward (Hinton 2022), feedback alignment (Lillicrap et al. 2016), local losses.
- **Measure:** memory use, and quality gap relative to backprop. Expect a quality gap: this is a hard research area, so frame the goal as memory savings or hardware friendliness, not beating backprop.

### Idea 7: Eligibility traces and three-factor updates (optimizer / learning rule)

- **Inspiration:** synapses keep a decaying trace of recent activity until a broadcast signal arrives.
- **Translation:** `Δw = η · e(t) · M(t)`, with `e` decaying as `de/dt = −e/τ + (pre × post)`. Useful for delayed credit assignment in recurrent models and RL.
- **Related:** eligibility traces in RL, e-prop (Bellec et al. 2020).
- **Measure:** performance on tasks with delayed reward, memory vs. backprop through time.

### Idea 8: Curriculum and developmental schedules (data)

- **Inspiration:** development proceeds from simple to complex, and low-detail input early on.
- **Translation:** order data from simple to hard, start at low resolution or short sequences and increase gradually.
- **Related:** curriculum learning (Bengio et al. 2009), progressive resizing, sequence-length warmup.
- **Measure:** steps to target loss with the same data and model.

### Idea 9: Dynamic sparse training (architecture / efficiency)

- **Inspiration:** synaptic pruning and growth.
- **Translation:** train a sparse network from the start, periodically removing weak connections and growing new ones.
- **Related:** SET, RigL, lottery-ticket research.
- **Measure:** FLOPs and memory for equal quality, ideally training cost (not only inference cost).

### Idea 10: Curiosity-driven data selection (data)

- **Inspiration:** the brain focuses on what is novel but learnable.
- **Translation:** prioritize training examples with high *learning progress* (loss that is high but dropping), skipping both trivial and unlearnable ones.
- **Related:** importance sampling, active learning, data pruning, hard-example mining.
- **Measure:** steps to target loss at equal data, or accuracy with a smaller dataset.

### Idea 11: Meta-learned initialization and priors (architecture)

- **Inspiration:** evolution gives newborns strong priors.
- **Translation:** meta-learn a good starting point or architecture so training needs fewer steps.
- **Related:** MAML (Finn et al. 2017), neural architecture search, learned optimizers.
- **Measure:** steps to adapt on new tasks. Often expensive at scale.

### Idea 12: Dendritic and richer units (architecture, speculative)

- **Inspiration:** single neurons perform nonlinear computation across dendritic branches.
- **Translation:** units with small internal structure (e.g. several sub-units with nonlinearities) might need fewer total units.
- **Related:** research on dendritic neural networks.
- **Measure:** quality per parameter and per FLOP. Speculative, with a high risk that gains vanish after matching compute.

### Suggested priority for a first paper

| Idea | Lever | Difficulty | Testable at small scale | Risk of "already done" |
|---|---|---|---|---|
| 1. Surprise-gated learning rate | Optimizer | Low | Yes | Medium |
| 2. Homeostatic update normalization | Optimizer | Low-Medium | Yes | Medium-High |
| 3. Metaplastic per-parameter rates | Optimizer | Medium | Yes | Medium |
| 4. Fast and slow weights | Optimizer | Low | Yes | High (Lookahead) |
| 5. Sparse activation | Architecture | Medium | Yes | High |
| 8. Curriculum schedules | Data | Low | Yes | High |
| 10. Curiosity-driven data selection | Data | Medium | Yes | Medium |
| 6. Local learning | Gradient | High | Partly | High |
| 9. Dynamic sparse training | Architecture | Medium | Yes | High |
| 11. Meta-learned initialization | Architecture | High | Partly | High |
| 12. Dendritic units | Architecture | High | Partly | Medium |

A sensible starting set: **Ideas 1, 2 and 3**, because they only change the optimizer, are cheap to test, and map directly to the Phase 1 plan in your proposal. Idea 3 is interesting because it connects *importance* (a brain concept) with the second-moment statistics already inside Adam, so there may be room for a clean formula.

---

## 9. How to turn an analogy into a research hypothesis

1. **Pick one biological principle** and state what problem it solves for the brain (e.g. "keep activity stable while learning").
2. **Ask whether models have that problem** (e.g. unstable training at high learning rates). If not, the analogy may not help.
3. **Write it as math** in the form from Section 7: local signal × modulating signal × stabilizing term.
4. **Implement it** as a small optimizer or layer, a few dozen lines.
5. **Check the literature** for the same mechanism under a different name.
6. **Define one metric** (steps to target loss, FLOPs, forgetting) *before* running experiments.
7. **Compare against tuned baselines** with multiple seeds and equal compute.
8. **Explain the mechanism:** use diagnostics (update norms, per-layer activity, loss curves) to show *why* it helps. Saying "inspired by the brain" is not an explanation.

Hypothesis template:

> Replacing [standard component] with [bio-inspired version] will reduce [metric] by [X%] on [model + dataset], because [mechanism], which in the brain serves to [biological function].

---

## 10. Cautions

- **Biological plausibility is not the goal.** The goal is a better training method. A mechanism can be less biological and more effective.
- **The brain is not necessarily optimal.** Evolution optimizes for survival under biological limits, and the brain has bugs, biases, and limits.
- **Our understanding is incomplete.** Claims like "the brain does X" are often hypotheses. Phrase them carefully ("is thought to," "one theory proposes").
- **Many bio-inspired methods fail to beat strong baselines.** This is normal, so treat negative results as data, and keep a log of them.
- **Do not over-claim in a paper.** Say "inspired by," not "mimics the brain," unless you have evidence.
- **Hardware differs.** Sparse or event-driven methods may save FLOPs on paper but not speed on GPUs. Measure wall-clock time too.

---

## 11. Suggested reading

**Brain learning and memory**

- Hebb, *The Organization of Behavior* (1949)
- Bienenstock, Cooper and Munro, "Theory for the development of neuron selectivity" (BCM, 1982)
- Bi and Poo, "Synaptic modifications in cultured hippocampal neurons" (STDP, 1998)
- Turrigiano, "The self-tuning neuron: synaptic scaling of excitatory synapses" (2008)
- McClelland, McNaughton and O'Reilly, "Why there are complementary learning systems in the hippocampus and neocortex" (1995)
- Schultz, Dayan and Montague, "A neural substrate of prediction and reward" (1997)
- Rao and Ballard, "Predictive coding in the visual cortex" (1999)
- Benna and Fusi, "Computational principles of synaptic memory consolidation" (2016)

**Bridging brain and machine learning**

- Marblestone, Wayne and Kording, "Toward an integration of deep learning and neuroscience" (2016)
- Hassabis, Kumaran, Summerfield and Botvinick, "Neuroscience-inspired artificial intelligence" (2017)
- Lillicrap, Santoro, Marris, Akerman and Hinton, "Backpropagation and the brain" (2020)
- Richards et al., "A deep learning framework for neuroscience" (2019)
- Whittington and Bogacz, "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity" (2017)
- Ramsauer et al., "Hopfield Networks is All You Need" (2020)

**Mechanisms as ML methods**

- Lillicrap et al., "Random synaptic feedback weights support error backpropagation for deep learning" (2016)
- Hinton, "The Forward-Forward Algorithm: Some Preliminary Investigations" (2022)
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC, 2017)
- Zenke, Poole and Ganguli, "Continual learning through synaptic intelligence" (2017)
- Bellec et al., "A solution to the learning dilemma for recurrent networks of spiking neurons" (e-prop, 2020)
- Zhang, Lucas, Hinton and Ba, "Lookahead Optimizer: k steps forward, 1 step back" (2019)
- Mocanu et al., "Scalable training of artificial neural networks with adaptive sparse connectivity" (SET, 2018) and Evci et al., "Rigging the Lottery" (RigL, 2020)
- Bengio et al., "Curriculum learning" (2009)
- Finn, Abbeel and Levine, "Model-Agnostic Meta-Learning" (2017)

Titles, years, and details above are from memory and may contain small errors, so look each one up before citing it. Also search for newer work, since both neuroscience-inspired ML and optimizer research move quickly.

---

## 12. Quick summary

- The brain and models differ most in **learning rule**, **sparsity and energy**, **memory organization**, **data efficiency**, and **continual learning**, more than in "analog vs. digital."
- The brain learns with **local rules shaped by broadcast signals**, stabilized by **homeostasis and metaplasticity**, using **fast and slow memory systems**, **replay**, **curiosity**, and **strong inherited priors**.
- Models learn with **global gradients**, a **single learning speed**, **dense computation**, and **frozen weights**.
- Each difference suggests a testable mechanism. The most accessible first experiments change the **optimizer**: surprise-gated learning rates, homeostatic normalization, and importance-aware per-parameter rates.
- Whatever the inspiration, the contribution counts only if it beats a **tuned baseline** under **equal compute** across **multiple seeds**, with an explanation of *why*.
