# Formal Definitions — Fuzzy Box-Embedding Framework

Companion to the paper *"Fuzzy Learning for Box Embeddings"* (Anonymous).

**Configuration used for all reported results (final, Nov 2026):** product
t-norm, Reichenbach implication, fixed temperature $\tau = 0.1$, learning
rate $0.05$, AdamW, 5,000 steps, batch 512, weights $w_\alpha = 2.0$,
$w_\beta = 2.0$, $w_\delta = 0.1$, $w_r = 0.1$, $w_{\mathrm{contact}} = 0.1$
(crisp margin tail on the **entailed** closure), no sibling term, plain
(non-curriculum) schedule. Sweep tree: `results_fuzzy_entailed/`.

---

## 1. Notation and Geometric Representation

A named OWL class $\mathcal{C}$ is embedded as an axis-aligned box in
$\mathbb{R}^{d}$, where $d$ is the embedding dimensionality. The box is
identified by a centre vector $\mu_{\mathcal{C}}$ and a half-size vector
$h_{\mathcal{C}} \in \mathbb{R}_{+}^{d}$. Its crisp extents are the
component-wise combinations

$$
\mathbf{m}_{\mathcal{C}} = \mu_{\mathcal{C}} - h_{\mathcal{C}}, \qquad
  \mathbf{M}_{\mathcal{C}} = \mu_{\mathcal{C}} + h_{\mathcal{C}},
$$

so the crisp box occupies the hyperrectangle
$[\mathbf{m}, \mathbf{M}]$ — the component-wise combinations above. Both
parameters are initialised from a seeded standard normal distribution scaled
by $0.1$,

$$
\mu_{\mathcal{C}}, h_{\mathcal{C}} \sim \mathcal{N}(0, 0.1^{2}),
$$

and the half-size is parametrised as $\mathrm{softplus}(\cdot) + 10^{-6}$, so
each side length is strictly positive and boxes cannot collapse to zero width.

Notation: $\sqsubseteq$ is the subclass relation, where $a \sqsubseteq b$ means
$a$ is a subclass of $b$; $\perp$ is the disjointness relation, where
$a \perp b$ means $a$ and $b$ are disjoint. The edge sets are:

- $\mathcal{E}_{\sqsubseteq}$ — closure edges of the subclass hierarchy
- $\mathcal{E}_{\perp}$ — entailed disjoint pairs
- $\mathcal{E}_{\perp}^{\mathrm{asserted}}$ — asserted disjoint axioms

with the asserted subset of the entailed pairs

$$
\mathcal{E}_{\perp}^{\mathrm{asserted}} \subseteq \mathcal{E}_{\perp}.
$$

## 2. Membership Function

Membership of a point $x$ in the box of class $\mathcal{C}$, per dimension $j$,
is the sigmoidal function

$$
\mu_{\mathcal{C}}^{(j)}(x) = \sigma\left( \frac{h_{\mathcal{C}}^{(j)} - |x - \mu_{\mathcal{C}}^{(j)}|}{\tau} \right),
$$

where $\sigma$ is the logistic function and $\tau \in [10^{-4}, 10]$ is the
temperature, fixed here at $\tau = 0.1$. Membership is $0.5$ exactly on the box
boundary, approaches $1$ in the interior, and approaches $0$ outside. The crisp
bounds $\mathbf{m}$ and $\mathbf{M}$ are used for geometric evaluation and for
the min-box-size regulariser.

## 3. T-norm and Implication

Conjunction uses the **product t-norm**,

$$
T(a, b) = a \cdot b,
$$

and containment uses the **Reichenbach implication**,

$$
I(a, b) = 1 - a + a \cdot b.
$$

## 4. Overlap Degree

For boxes $a$ and $b$, membership is evaluated at the midpoints and both
centres (three probe points per dimension),

$$
\mathcal{P}_{ab} = \lbrace \frac{\mu_a + \mu_b}{2}, \mu_a, \mu_b \rbrace,
$$

and the overlap degree aggregates the t-norm as the minimum over dimensions of
the maximum over probe points,

$$
\mathrm{overlap}(a, b) = \min_{1 \le j \le d} \max_{x \in \mathcal{P}_{ab}} T\big( \mu_a^{(j)}(x), \mu_b^{(j)}(x) \big).
$$

A single overlapping point suffices for an overlap. Because the probe points may
miss the point of maximum overlap, the fuzzy quantity is an approximation; the
geometric margin metric is reported as ground truth.

## 5. Containment Degree

For a subclass pair $(c, p)$, membership is evaluated at the child's centre and
the centres of its $2d$ faces,

$$
\mathcal{X}_c = \lbrace \mu_c \rbrace \cup \lbrace \mu_c \pm h_c^{(j)} : j = 1, \dots, d \rbrace,
$$

and the containment degree is the weakest implication between child and parent
membership over all probe points and dimensions,

$$
\mathrm{contain}(c, p) = \min_{x \in \mathcal{X}_c} \min_{1 \le j \le d} I\big( \mu_c^{(j)}(x), \mu_p^{(j)}(x) \big).
$$

## 6. Fuzzy Cardinality

The oversize regularisation uses the log fuzzy cardinality, an approximation of
the per-dimension integral of the sigmoidal membership,

$$
\mathrm{card}_{\log}(\mathcal{C}) = \sum_{j=1}^{d} \log\big( 2 h_{\mathcal{C}}^{(j)} + \pi \tau \big).
$$

## 7. Composite Loss

The training objective is a weighted sum of four differentiable terms — two
axiom constraints and two structural regularisers:

$$
\mathcal{L} = \underbrace{w_{\alpha} \alpha + w_{\beta} \beta}_{\text{axiom constraints}} + \underbrace{w_{\delta} \delta + w_r r}_{\text{structural regularisers}},
$$

with $w_\alpha = 2.0$, $w_\beta = 2.0$, $w_\delta = 0.1$, $w_r = 0.1$ and
contact weight $w_{\mathrm{contact}} = 0.1$. Each term is a mean over its
respective axiom set, keeping the composite loss scale-comparable across
ontologies of different sizes.

### 7.1 Subclass containment ($\alpha$, $L_\mathrm{subclass}$)

$$
L_{\mathrm{subclass}} = \frac{1}{\lvert \mathcal{E}_{\sqsubseteq} \rvert}
  \sum_{(c,p) \in \mathcal{E}_{\sqsubseteq}}
  \log\big( 1 + \lvert \mathrm{desc}(c) \rvert \cdot \lvert \mathrm{anc}(p) \rvert \big)
  \cdot \big( 1 - \mathrm{contain}(c,p) \big),
$$

where $\lvert \mathrm{desc}(c) \rvert$ and $\lvert \mathrm{anc}(p) \rvert$ are the descendant and
ancestor counts of child and parent. The logarithm dampens the raw entailment
count so that a few long-range axioms do not dominate the gradient; the term
vanishes only when containment reaches one.

### 7.2 Disjointness ($\beta$, $L_\mathrm{disjoint}$)

$$
L_{\mathrm{disjoint}} = \frac{1}{\lvert \mathcal{E}_{\perp} \rvert}
  \sum_{(a,b) \in \mathcal{E}_{\perp}} \mathrm{overlap}(a,b) +
  w_{\mathrm{contact}} \cdot \frac{1}{\lvert \mathcal{E}_{\perp} \rvert}
  \sum_{(a,b) \in \mathcal{E}_{\perp}}
  \mathrm{softplus}\big( \mathrm{margin} - \mathrm{sep}(a,b) \big),
$$

with the crisp margin of separation across dimensions

$$
\mathrm{sep}(a,b) = \max_{1 \le j \le d} \max\big( mn_b^{(j)} - mx_a^{(j)}, mn_a^{(j)} - mx_b^{(j)} \big),
$$

and $\mathrm{margin} = 0.02$ (matches the geometric evaluation margin). The fuzzy
overlap term alone cannot hold disjoint boxes apart once they barely touch (its
gradient dies out at a separation of roughly $3\tau$); the crisp contact tail
therefore arbitrates the **entailed** pairs in the margin-based manner of the
penalty baseline. Applied to the entailed closure at a low weight
($w_{\mathrm{contact}} = 0.1$), it removes the remaining seed-dependent
disjointness residuals at negligible containment cost; higher weights
($\ge 0.25$) over-shove descendant boxes and break containment.

### 7.3 Oversize regularisation ($\delta$, $L_\mathrm{oversize}$)

$$
L_{\mathrm{oversize}} = \frac{1}{n} \sum_{\mathcal{C}} \Big[
  \mathrm{softplus}\big( \mathrm{card}_{\log}(\mathcal{C}) - t(\mathcal{C}) \big) +
  \mathrm{softplus}\big( \text{min-log-volume} - \mathrm{card}_{\log}(\mathcal{C}) \big) \Big],
$$

with the depth-dependent target

$$
t(\mathcal{C}) = \max\Big( \text{min-log-volume},
  \text{base-log-volume} - \text{depth-scale} \cdot \sqrt{\mathrm{depth}(\mathcal{C})} \Big),
$$

where $\mathrm{depth}(\mathcal{C})$ is the graph distance of the class from a root.
The first term penalises boxes above the target; the second prevents collapse
to zero volume (anti-degeneracy).

### 7.4 Min-box floor ($r$)

$$
L_{\text{min-box}} = \frac{1}{nd} \sum_{\mathcal{C}} \sum_{j=1}^{d}
  \mathrm{softplus}\left( \text{min-box-size} - 2h_{\mathcal{C}}^{(j)} \right),
$$

with $\mathrm{softplus}(x) = \ln(1 + e^{x})$ and a fixed min-box size of
$0.05$. Prevents individual box widths from collapsing.

## 8. Evaluation Criteria

**Geometric checks.** A subclass edge $(c, p)$ is violated if the child's crisp
extents fall outside the parent's in any dimension ($m_c^{(j)} < m_p^{(j)}$ or
$M_c^{(j)} > M_p^{(j)}$ for some $j$). A disjoint pair is violated if the maximum
margin of separation across dimensions is below the margin,
$\max_{j} \mathrm{sep}^{(j)}(a,b) < 0.02$.

**Fuzzy graded satisfaction.** Subclass satisfaction is $\mathrm{contain}(c,p)$;
disjoint satisfaction is $1 - \mathrm{overlap}(a,b)$. Violations are counted at
thresholds $\alpha \in \lbrace 0.9, 0.95, 0.99 \rbrace$; each threshold additionally reports
the average violation degree (the mean $1 - \mathrm{satisfaction}$ over violated
edges), separating "barely missed" from "catastrophically wrong". Metrics are
reported on the transitive closure and on the asserted axioms separately.

## 9. Training Configuration

5,000 steps, AdamW, learning rate $0.05$, weight decay $10^{-4}$, gradient
clipping at norm $1.0$; losses evaluated over all extracted axioms in batches of
512 edges. All terms active throughout training (plain schedule); curriculum
schedules were evaluated and consistently degraded final performance.

## 10. Seed-42 Dimension Sweep (Supplementary)

Geometric violation rates at seed 42 under the final configuration
(`results_fuzzy_entailed/`, Nov 2026), reported as *subclass/disjointness* in
percent. Disjointness is not applicable to CIM (no disjointness axioms).

| Ontology | d2 | d4 | d6 | d8 | d10 | d15 | d20 | d25 | d30 |
|---|---|---|---|---|---|---|---|---|---|
| CIM  | 0.00/– | 0.00/– | 0.00/– | 0.00/– | 0.00/– | 0.29/– | 2.22/– | 4.11/– | 5.67/– |
| GO   | 0.25/0.00 | 0.86/0.00 | 0.25/0.00 | 0.27/0.00 | 2.59/0.00 | 3.49/0.00 | 5.25/0.00 | 6.95/0.00 | 6.53/0.00 |
| DOID | 0.72/0.00 | 0.04/0.00 | 2.95/0.00 | 3.03/0.00 | 3.12/0.00 | 3.89/0.00 | 5.18/0.00 | 6.34/0.00 | 7.13/0.00 |
| OEO  | 2.46/0.15 | 4.16/0.02 | 1.14/0.00 | 4.71/0.00 | 5.64/0.00 | 8.97/0.00 | 11.46/0.00 | 13.79/0.00 | 14.21/0.00 |

The OEO odd dimensions 3–11 complete the picture: 0.97 (d=3), 4.26 (d=5),
6.06 (d=7), 4.94 (d=9), 7.42 (d=11) % subclass violations, all with
disjointness 0.00 % except 0.02 % at d=3. Under the final configuration the
disjointness spikes of the superseded baseline no longer occur: every
disjointness cell above is zero except the isolated residuals at OEO d2 (≤ 8 of
4,606 entailed pairs) and d4/d6 (single pairs each).

## 11. Known Approximations (documented, not defects)

1. **Fuzzy cardinality** uses $2h + \pi\tau$ per dimension; the exact
   integral of the sigmoid is $2h + 2\tau\ln(1 + e^{-h/\tau})$, so the
   approximation overestimates by up to a factor ≈2.3 when $h \ll \tau$.
   At trained box sizes the error is 8–30 % and is absorbed by the calibrated
   weights.
2. **Overlap probing** at three points can under- or over-estimate the true
   fuzzy overlap, since the per-dimension maxima may combine coordinates of
   different probe points. The geometric margin metric is the authoritative
   correctness check.
3. **Entailment weighting** is an unnormalised weighted mean, which scales the
   subclass loss by the mean entailment weight — the effective
   $w_\alpha$ is larger than its nominal value. The scaling is identical across
   runs, so cross-run comparisons are unaffected.
