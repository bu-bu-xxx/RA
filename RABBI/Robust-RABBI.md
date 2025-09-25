## 1) Parameters & Objects used by the algorithm

### Core instance & environment state

- **Products & resources.**
    `n` (int): number of products. `d` (int): number of resource types. $A \in \mathbb{R}^{n \times d}$: resource usage; row $i$ is the usage vector for product $i$ (nonnegative). $B \in \mathbb{R}^d$: initial resource budgets. $b \in \mathbb{R}^d$: current remaining budgets (state).
- **Time.**
    `T` (int): horizon; $t$ (int): current period index ($t \in \{0, \dots, T-1\}$). The remaining time is $T-t$.
- **Price grids & actions (menus).**
    For each product $i$, a finite price set $f_{\text{split}}[i] = P_i = \{p_{i1}, \dots, p_{i, |P_i|}\}$. A **menu/action** is a vector $\alpha = (\alpha_1, \dots, \alpha_n)$ picking one price from each $P_i$. The (potentially huge) action space is $\mathcal{A} = \prod_i P_i$ with size $m = \prod_i |P_i|$. In code you may index actions by $\alpha \in \{0, \dots, m-1\}$ and store the associated price vector as the $\alpha$‑th column of $f \in \mathbb{R}^{n \times m}$. (Each row of `f_split` must include a large enough “no‑sell” price.)
- **Demand model.**
    `demand_model` $\in$ `{MNL, LINEAR}` with its parameters (e.g., `MNL.d, MNL.mu, MNL.u0, MNL.gamma` or `Linear.psi, Linear.theta`). For any menu $\alpha$, the purchase probabilities are $p_{i,\alpha} = q_i(\alpha) \in [0,1]$ with $\sum_i p_{i,\alpha} \le 1$ (the remainder is the outside option). `tolerance` zeroes tiny probabilities. (If you use the OFFline benchmark you may take $Q[t,i,\alpha]$ as the probability tensor at time‑to‑go $t$.)
- **Simulator data (optional).**
    $Y[t,\alpha] \in \{-1, 0, \dots, n-1\}$: pre‑sampled realized choice given menu $\alpha$ at time $t$ ($-1$ means no purchase). If not using `Y`, you can sample from $p_{\cdot, \alpha}$ on the fly. Histories: `b_history, alpha_history, j_history, reward_history`.

### LP primitives derived per action $\alpha$

- **Per‑action revenue coefficient**

$$
r_\alpha = \sum_{i=1}^n f_{i,\alpha}\,p_{i,\alpha}.
$$

- **Per‑action resource coefficients** (for resource $k=1..d$)

$$
c_{k,\alpha} = \sum_{i=1}^n A_{i,k}\,p_{i,\alpha}.
$$

- **Time coefficient:** each action consumes one unit of remaining time, so coefficient is $1$.
    These are exactly the coefficients used in your “Core LP” at time $t$: maximize $\sum_\alpha x_\alpha r_\alpha$ subject to $\sum_\alpha x_\alpha c_{k,\alpha} \le b_k$ and $\sum_\alpha x_\alpha = T-t$, $x_\alpha \ge 0$.

### Robust‑RABBI Step‑2 (dual) objects

- **Dual variables:** $\lambda \in \mathbb{R}_+^d$ (resource prices), $\mu \in \mathbb{R}_+$ (time price).

- **Dual constraint for action $\alpha$:**

$$
\sum_{k=1}^d c_{k,\alpha}\,\lambda_k + \mu \ge r_\alpha.
$$

The **reduced cost** (violation) at $(\lambda,\mu)$ is
$$
\bar{c}_\alpha(\lambda,\mu) = r_\alpha - \sum_{k=1}^d c_{k,\alpha}\,\lambda_k - \mu = \sum_{i=1}^n p_{i,\alpha}\left(f_{i,\alpha} - A_{i,:}^{\top}\lambda\right) - \mu.
$$

- **Separation objective:**
    $V^*(\lambda,\mu) = \max_{\alpha \in \mathcal{A}} \bar{c}_\alpha(\lambda,\mu)$.
     The (approximate) oracle returns $\tilde{V}(\lambda,\mu) \ge (1-\varepsilon_{\text{sep}})V^*(\lambda,\mu)$ and the corresponding top‑violating menu(s).

- **Stopping rule:** stop ellipsoid when $\tilde{V}(\lambda,\mu) \le \eta$; this implies $V^* \le \eta/(1-\varepsilon_{\text{sep}})$.

- **Generated column set:** $\mathcal{A}'_t$ is the set of violating menus discovered up to termination at time $t$ (you may add top‑$k$ violators per iteration for speed).
    (All of these are the specialized forms of the generic dual/separation described in your parameters note.)

------

## 2) Textual Algorithm (main + sub‑algorithms with formulas)

We describe **Robust‑RABBI (single‑type, multi‑product)** at period $t$ with remaining budget $b$.

### Main loop (one period)

**Input:** $A, f$ (or `f_split`), demand model, $T, b, t$, tolerance, and solver tolerances $\varepsilon_{\text{sep}} > 0$ (oracle accuracy) and $\eta > 0$ (feasibility slack).
**Output:** chosen menu $\alpha_t$, realized buy $j_t \in \{-1, 0, \dots, n-1\}$, reward, and updated $b$.

1. **Build feasibility‑aware menu space for period $t$.**
    For any product $i$ with **insufficient** inventory to fulfill a sale (i.e., $b$ cannot cover row $A_{i,:}$), **force** $\alpha_i$ to the “no‑sell” price (i.e. a large-enough price in price_set_matrix) in $P_i$. This keeps any offered menu feasible under instantaneous inventory. (Practically, we apply a no-sell price trick: if $b$ cannot support product $i$, we set a large-enough price for product $i$ and the large-enough price is guaranteed in the `price_set_matrix[i: -1]`)

2. **Resolve (estimate)** demand & coefficients.
    Using the demand model, compute (or lazily evaluate inside the oracle) $p_{i,\alpha}=q_i(\alpha)$ and thus

$$
 r_\alpha = \sum_i f_{i,\alpha}p_{i,\alpha}, \qquad c_{k,\alpha} = \sum_i A_{i,k}p_{i,\alpha}.
$$

 The remaining time is $T-t$.

3. **Step‑2: Dual resolution by ellipsoid + approximate separation.**
    Initialize an ellipsoid over $(\lambda,\mu) \in [0, f_{\max}]^d \times [0, f_{\max}]$. Repeat:

 - Query the separation oracle with current $(\lambda',\mu')$ to get $\tilde{V}$ and top‑violating menu(s) $\tilde{\alpha}$ achieving

$$
 \tilde{V}(\lambda',\mu') \ge (1-\varepsilon_{\text{sep}})\,\max_{\alpha} \left\{r_\alpha - \sum_k c_{k,\alpha}\lambda'_k - \mu'\right\}.
$$

 - If $\tilde{V}(\lambda',\mu') > \eta$: **add** $\tilde\alpha$ to $\mathcal{A}'_t$ and cut the ellipsoid with the violated dual constraint for $\tilde\alpha$.

 - Else (no significant violation): **terminate**.
    This yields a small column set $\mathcal{A}'_t$ and a near‑feasible dual point (all reduced costs $\le \tau_t = \eta/(1-\varepsilon_{\text{sep}})$).

4. **Step‑3: Solve the restricted primal (scores).**
    Solve the LP over the discovered columns only:

$$
 \begin{aligned}
 \max_{x \ge 0} \quad & \sum_{\alpha \in \mathcal{A}'_t} x_\alpha\, r_\alpha \\
 \text{s.t.} \quad & \sum_{\alpha \in \mathcal{A}'_t} x_\alpha\, c_{k,\alpha} \le b_k \quad (k=1,\dots,d), \\
 & \sum_{\alpha \in \mathcal{A}'_t} x_\alpha = T-t.
 \end{aligned}
$$

 The optimizer $x^{(t)}$ gives **scores**; let $\mathrm{score}(\alpha) = x^{(t)}_\alpha$ for $\alpha \in \mathcal{A}'_t$ and 0 otherwise. (This is the “action summaries” step your core LP already implements.)

5. **Act (choose the menu).**
    Among feasible menus w.r.t. current inventory (after Step 1), choose

$$
 \alpha_t \in \arg\max_{\alpha} \mathrm{score}(\alpha).
$$

6. **Simulate outcome and update.**
    Draw $j_t$ from $Y[t, \alpha_t]$ if pre‑generated; otherwise sample from $p_{\cdot, \alpha_t}$.
     If $j_t \ge 0$: receive reward $f_{j_t, \alpha_t}$ and update $b \leftarrow b - A_{j_t,:}$; else reward $=0$.
     Record histories and continue with $t \leftarrow t+1$.

------

### Sub‑algorithms

**(S1) Separation‑oracle objective (single‑type specialization).**
Given $(\lambda,\mu)$ and a menu $\alpha$,
$$
\bar{c}_\alpha(\lambda,\mu) = \underbrace{\sum_i f_{i,\alpha}\,q_i(\alpha)}_{r_\alpha} - \underbrace{\sum_k \left(\sum_i A_{i,k}\,q_i(\alpha)\right)\lambda_k}_{c_{\cdot,\alpha}^{\top}\lambda} - \mu.
$$
Return the maximizer(s) over $\alpha \in \mathcal{A}$ together with the value. (Implement via enumeration when $m$ is small; otherwise use the problem‑specific FPTAS/heuristic search over the menu lattice.)

**(S2) Ellipsoid update & stopping.**
Stop when $\tilde{V} \le \eta$. Otherwise cut with the hyperplane for $\tilde\alpha$:
$$
\sum_k c_{k,\tilde{\alpha}}\,\lambda_k + \mu \ge r_{\tilde{\alpha}}.
$$

**(S3) Restricted primal LP.**
As above; you already have this “Core LP” implemented (use only columns in $\mathcal{A}'_t$).

**(S4) Feasibility filter for menus.**
If $b$ cannot cover product $i$ (i.e., any coordinate of $A_{i,:}$ exceeds $b$ when purchased), force menu’s price for $i$ to the “no‑sell” value or set $p_{i,\alpha}=0$ before evaluating $r_\alpha, c_{\cdot,\alpha}$.

------

## 3) Pseudocode (mirrors the textual algorithm)

```text
Algorithm Robust-RABBI (Single-Type Multi-Product)

Inputs:
  A ∈ ℝ^{n×d}, B ∈ ℝ^d, T, demand_model, f_split (or f),
  tolerance, ε_sep > 0, η > 0, topk ≥ 1 (optional)
State:
  b ← B;  t ← 0
Histories:
  init b_history = [B], alpha_history = [], j_history = [], reward_history = []

while t < T:

  # 1) Feasibility-aware menu space at period t
  define FeasibleMenu(α, b):
      for each product i:
          if not CanFulfill(b, A[i,:]):  # inventory check for product i
              force α_i to NO_SELL price in P_i (or set p_{i,α} = 0)
      return α

  # 2) Demand & coefficients (lazy if using oracle)
  define Coeffs(α):
      α ← FeasibleMenu(α, b)
      p_{iα} ← DemandProb(i, α, demand_model, tolerance) for i = 1..n
      r_α  ← ∑_i f_{i,α} * p_{iα}
      c_{kα} ← ∑_i A[i,k] * p_{iα} for k = 1..d
      return (r_α, c_{·α}, p_{·α})

  # 3) Step-2: Ellipsoid + approximate separation
  initialize ellipsoid E over (λ ∈ [0,f_max]^d, μ ∈ [0,f_max])
  𝒜'_t ← ∅
  repeat
      (λ', μ') ← center(E)

      # Separation oracle (S1): compute up to topk violating menus
      obtain topk menus (α̃₁,…,α̃_K) and value Ṽ s.t.
          Ṽ ≥ (1 - ε_sep) * max_α { r_α - c_{·α}^T λ' - μ' }
      if Ṽ > η:
          for each α̃ in {α̃₁,…,α̃_K}:
              𝒜'_t ← 𝒜'_t ∪ {α̃}
              add cut: c_{·α̃}^T λ + μ ≥ r_α̃
          E ← EllipsoidCut(E, all cuts added this iteration)
      else:
          break
  until convergence

  # 4) Step-3: Restricted primal LP (scores)
  build LP over columns in 𝒜'_t:
      maximize   ∑_{α∈𝒜'_t} x_α r_α
      s.t.       ∑_{α∈𝒜'_t} x_α c_{kα} ≤ b_k     for k=1..d
                 ∑_{α∈𝒜'_t} x_α = T - t
                 x_α ≥ 0
  solve LP → x^(t)
  define score(α) = x^(t)_α if α∈𝒜'_t else 0

  # 5) Act: choose menu with largest score among feasible ones
  α_t ← argmax_{α feasible under b} score(α)

  # 6) Simulate outcome and update
  j_t ← draw from Y[t, α_t] if available else sample from p_{·,α_t}
  if j_t ≥ 0:
      reward ← f_{j_t, α_t}
      b ← b - A[j_t, :]
  else:
      reward ← 0

  # Record and advance time
  append to histories: b, α_t, j_t, reward
  t ← t + 1

return histories
```

**Subroutines**

```text
function DemandProb(i, α, demand_model, tolerance):
    if demand_model == "MNL":
        # use d, mu, u0, gamma from YAML to compute q_i(α); clip by tolerance
    else if demand_model == "LINEAR":
        # p = ψ + Θ @ prices(α); clip and project so ∑_i p_i ≤ 1
    return p_{iα}

function CanFulfill(b, A_i):
    return all(b_k ≥ A_i[k] for k = 1..d)

function SeparationOracle(λ, μ, topk):
    # If m is small: enumerate all α ∈ 𝒜 and compute r_α, c_{·α}, take topk by violation
    # If m is large: call FPTAS/heuristic to search the menu lattice for top violators
    return {α̃₁,…,α̃_K}, Ṽ

function BuildColumn(α):
    (r_α, c_{·α}, p_{·α}) ← Coeffs(α)
    return r_α, c_{·α}
```

> **Notes for implementation.**
> • If `m` is modest, you can precompute $p[:,\alpha], r_\alpha, c_{\cdot,\alpha}$ once and index them in the oracle; otherwise evaluate on demand.
> • The restricted LP coincides with your “Core LP”, simply constrained to the columns discovered in the current period; you can solve it with SciPy HiGHS just like in your existing pipeline.
> • The feasibility filter (Step 1 / S4) is essential in the single‑type multi‑product case: if inventory cannot support product $i$, force its price to the “no‑sell” level (this matches your price‑grid convention).

**REMIND:** 

1. In the Robust-RABBI algorithm solver, we may run with a assortment set with a smaller size than `m` , but in the output of this solver, we need to ensure the parameters (especially `xxx_history` parameters) should maintain the same dimension as other solver (like RABBI, OFFline), (for `x_history` parameter as an example, we need to fill the infeasible product price by 0 since the sold probability of the infeasible product is 0)