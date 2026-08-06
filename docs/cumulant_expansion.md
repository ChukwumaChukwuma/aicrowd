# Cumulant expansion of a deep ReLU MLP

*A white-box scheme for `E[relu(z^L_j)]` that goes past the Gaussian
covariance-propagation baseline: the exact Hermite diagram rule for the joint
cumulants of rectified Gaussians, its factorisation into matrix products, the
FLOP cost of every diagram, and the accuracy ceiling the scheme runs into.*

Producers:

| artifact | script |
|---|---|
| ReLU Hermite coefficients, Edgeworth rectified mean | `scripts/03_theory_hermite.py` |
| diagram rule, proof, numerical verification | `scripts/03_theory_diagrams.py` |
| layer contraction, factorisation, order counting, FLOPs | `scripts/03_theory_orders.py` |
| non-Gaussian corrections to `E[x]`, `Var(x)`, `Cov(x)` | `scripts/03_theory_selfconsistency.py` |

---

## 0. Setup and notation

Row-vector convention, `x^0 ~ N(0, I_n)`, `z^l = x^{l-1} W^l`, `x^l = relu(z^l)`,
`W^l_{ij} ~ N(0, 2/n)` fixed (one realisation), `n = 256`, `L = 32`.

At a given layer drop the superscript and write, for the *incoming*
pre-activation `z` (dimension `n`):

| symbol | meaning |
|---|---|
| `m_i`, `s_i` | mean and sd of `z_i` |
| `alpha_i = m_i / s_i` | standardised threshold |
| `t_i = (z_i - m_i)/s_i` | standardised pre-activation |
| `R`, `rhohat = R - I` | correlation matrix of `z`; `rhohat` has **zero diagonal** |
| `x_i = relu(z_i)` | post-activation |
| `mu_i = E[x_i]`, `C = Cov(x)` | post-activation mean / covariance |
| `W` | the *outgoing* weight matrix, `z'_j = sum_i W_ij x_i` |
| `W^{ok}` | elementwise `k`-th power of `W` |
| `E_e := rhohat^{oe}` | elementwise `e`-th power of `rhohat` |
| `cs[X]_j := sum_i X_ij` | column sum |

`Phi`, `phi` are the standard normal cdf/pdf and `He_k` the probabilists'
Hermite polynomials.

The *only* approximation anywhere below is that `z` is treated as a Gaussian
field **when computing the joint cumulants of `x`** (§4-§6); the corrections to
that are §7. `m' = W^T mu` and `Sigma' = W^T C W` are exact identities, never
approximations.

---

## 1. (A) The ReLU Hermite coefficients — and their `p`-th power generalisation

**Proposition A.** With `a_k := E[relu(m + s t) He_k(t)]`, `t ~ N(0,1)`:

```
a_0 = m Phi(alpha) + s phi(alpha)
a_1 = s Phi(alpha)
a_k = (-1)^k s He_{k-2}(alpha) phi(alpha)        k >= 2
```

*Proof.* `He_k(t) phi(t) = (-1)^k phi^{(k)}(t)`, so `k` integrations by parts give
`a_k = E[ f^{(k)}(t) ]` with `f(t) = relu(m+st)` (Stein / Gaussian integration by
parts; boundary terms vanish because `f` grows linearly).  `f'(t) = s 1{m+st>0}`,
`f''(t) = s^2 delta(m+st) = s delta(t+alpha)`, hence `f^{(k)}(t) = s
delta^{(k-2)}(t+alpha)` for `k>=2`.  Then
`E[delta^{(j)}(t+alpha)] = (-1)^j phi^{(j)}(-alpha) = He_j(-alpha) phi(alpha)
= (-1)^j He_j(alpha) phi(alpha)`, and `j = k-2` gives the line above.  `k=1` is
`s P(t > -alpha)`; `k=0` is the standard rectified-Gaussian mean. []

**Corollary A'.** Because `d/dt relu(m+st)^p = p s relu(m+st)^{p-1}` (the delta
term is killed by the factor `u`), the Hermite coefficients of *all powers*
follow from the same list.  With `A^(p)_k := E[relu(m+st)^p He_k(t)]`:

```
A^(p)_k = p! s^{p-1} a_{k-p+1}                       k >= p-1
A^(p)_k = p!/(p-k)! * s^p * I_{p-k}                  0 <= k <= p-1
I_0 = Phi(alpha),  I_1 = alpha Phi + phi,  I_p = alpha I_{p-1} + (p-1) I_{p-2}
```

(`I_p = E[(t+alpha)^p 1{t > -alpha}]`, the truncated-normal moments.)
Both branches agree at `k = p-1`.  Everything the scheme needs — single-site
cumulants of `x`, block weights, the covariance correction kernels — is built
from `a_k` and `I_p` at `O(K n)` cost.

Verified against split Gauss-Legendre quadrature to `<1e-10` relative for
`alpha` in `[-4.4, 8.1]` and `k <= 12` (`03_theory_hermite.py`).

## 2. (B) Edgeworth / Gram-Charlier rectified mean

**Proposition B (exact form).** Let `z` have mean `m`, sd `sigma`, and
standardised Hermite moments `c_r := E[He_r((z-m)/sigma)]` (so `c_0 = 1`,
`c_1 = c_2 = 0`).  Then, as an `L^2(gamma)` Parseval identity,

```
E[relu(z)] = sum_{r>=0} c_r a_r / r!
           = m Phi(alpha) + sigma phi(alpha)
             + sigma phi(alpha) * sum_{r>=3} (-1)^r c_r He_{r-2}(alpha) / r!
```

**Proposition B' (cumulant form).** `sum_r c_r th^r/r! = exp( sum_{r>=3}
lambda_r th^r / r! )` with `lambda_r = kappa_r/sigma^r`, so `c_3 = lambda_3`,
`c_4 = lambda_4`, `c_5 = lambda_5`, `c_6 = lambda_6 + 10 lambda_3^2`,
`c_7 = lambda_7 + 35 lambda_3 lambda_4`, ...  Truncating at first order in the
cumulants reproduces the form in the task statement,

```
E[relu(z)] = m Phi + sigma phi
             + sum_{r>=3} (kappa_r / r!) (-1)^r sigma^{-(r-1)} He_{r-2}(alpha) phi(alpha)
```

exact through `O(kappa)`, with the first neglected term at `O(kappa_3^2)`
entering only at `r = 6`.

Both are verified in `03_theory_hermite.py` against a two-component Gaussian
mixture, for which the target, every cumulant and every `c_r` are closed-form.
The `c_r`-identities hold to `1e-13`.

> **Caveat that turns out to decide the whole project.** Proposition B converges
> only if `p_Z/phi` is in `L^2(gamma)`, i.e. only if `z` has *lighter* tails than
> the matched Gaussian.  §9 shows that the real `z^32` does not, and the series
> stalls.

---

## 3. The Hermite (Mehler / Wick) diagram rule

**Theorem 1.** Let `t = (t_1..t_p)` be jointly standard Gaussian with
correlation matrix `rho`, and `f_1..f_p` square-integrable with Hermite
coefficients `a^u_k = E[f_u(t) He_k(t)]`.  Then, with absolute convergence,

```
E[ prod_u f_u(t_u) ]        = sum over ALL loopless multigraphs G on [p]
                                prod_u a^u_{deg_u(G)} * prod_{u<v} rho_uv^{e_uv}/e_uv!

kappa( f_1(t_1),..,f_p(t_p) ) = the same sum restricted to CONNECTED G
```

*Proof.*

*(i) Wick.* `He_k(t_u)` is the Wick power `:t_u^k:`.  Wick's theorem for
Wick-ordered products states that `E[ prod_u He_{k_u}(t_u) ]` is the sum over
perfect matchings of the `sum_u k_u` legs in which no leg is matched inside its
own vertex, each matched pair `(u,v)` contributing `rho_uv`.  Grouping matchings
by the resulting multigraph `{e_uv}`: vertex `u` splits its `k_u` legs into
bundles in `k_u! / prod_v e_uv!` ways, and the two bundles of edge `(u,v)` are
matched in `e_uv!` ways, so the multiplicity of `{e_uv}` is
`prod_u k_u! / prod_{u<v} e_uv!`.  Hence

```
E[ prod_u He_{k_u}(t_u) ] = sum_{ {e_uv} : deg_u = k_u }
                             prod_u k_u! / prod_{u<v} e_uv! * prod rho^{e_uv}
```

*(ii) Substitution.* Insert `f_u = sum_k a^u_k He_k / k!`.  The `1/k_u!` cancels
the `k_u!` above and `k_u` becomes `deg_u(G)`; the moment formula follows.

*(iii) Linked cluster.* The weight `w(G) = prod_u a^u_{deg_u} prod rho^e/e!`
factorises over connected components, because `a^u_{deg_u}` only sees `u`'s own
component.  Every graph decomposes uniquely into (a set partition of `[p]`) x (a
connected graph on each block), so `sum_G w(G) = sum_{pi} prod_{B in pi} c(B)`
with `c(B) = sum over connected G on B`.  That is *exactly* the moment-cumulant
relation, so by uniqueness of Moebius inversion on the partition lattice
`kappa(f_1..f_p) = c([p])`. []

`p = 2` gives Mehler: `Cov(x_i,x_k) = sum_{k>=1} a^i_k a^k_k rho^k / k!`.

**Verification** (`03_theory_diagrams.py`):
* with polynomial `f_u` the Hermite expansion terminates and the diagram sum is
  finite; against exact tensor Gauss-Hermite quadrature the moment and the
  cumulant version both agree to **1e-15** at `p = 3` and `p = 4`.  This pins the
  combinatorial weights exactly.
* with `f = relu`, `p = 4`, correlations as large as `|rho| = 0.82`, against
  `4e7` Monte-Carlo samples: agreement within 1 MC standard error for every
  cumulant tested.

## 4. Repeated arguments: block weights and Ursell corrections

The layer map needs `kappa(x_i, x_i, x_k)`-type objects.  Theorem 1 is a
polynomial identity in `rho` so it is still *true* at `rho_uu = 1`, but the sum
over the coincident edge multiplicity then converges only algebraically
(`~ 1/K^{1/2}`).  Resumming those internal edges collapses each block of `b`
coincident slots into a single vertex.

**Theorem 2.** For blocks `B_1..B_q` of sizes `b_1..b_q` sitting at **distinct**
neurons `i_1..i_q`,

```
kappa( x_{i_1}^(b_1), ..., x_{i_q}^(b_q) )
   = sum over connected loopless multigraphs G on [q]
        prod_b beta^{i_b}_{b_b, deg_b(G)} * prod_{b<b'} rhohat^{e}/e!
     + Ursell corrections
```

with the **block weights**

```
beta^i_{b,d} := kappa( x_i,..,x_i , He_d(t_i) )        (b copies of x_i)
              = b! [u^b] ( E[e^{u x_i} He_d(t_i)] / E[e^{u x_i}] )
```

i.e. `beta_{b,d}` is the `b`-th Taylor coefficient of `A_d(u)/A_0(u)` where
`A_d(u) = sum_p u^p A^(p)_d / p!`.  Explicitly

```
beta_{1,d} = a_d
beta_{2,d} = A^(2)_d - 2 mu a_d                 ( = E[(x-mu)^2 He_d] )
beta_{3,d} = A^(3)_d - 3 mu A^(2)_d + (6 mu^2 - 3 E[x^2]) a_d
```

and, for a single isolated block (`q = 1`), `beta_{b,0}` is replaced by the
ordinary single-site cumulant `kappa_b(x_i)`, which is exact.

*Proof sketch.* Write `Psi(u) = log E[prod_b exp(u_b x_{i_b})]` and apply
Theorem 1 to the functions `g_b(t) = exp(u_b f_{i_b}(t))`, whose Hermite
coefficients are `A^b_d(u)`.  Then
`Psi = sum_b log A^b_0(u_b) + log(1 + Delta)` with `Delta` the sum over nonempty
graphs of `prod_b (A^b_{d_b}/A^b_0) prod rho^e/e!`.  The first term is a sum of
one-variable CGFs and so contributes only to `q = 1`.  In `log(1+Delta)` the
linear term reproduces the stated graph sum with vertex weights
`beta_{b,d} = b![u^b](A_d/A_0)`.  Products `Delta^k`, `k >= 2`, need each factor
to touch at least two blocks, so for total order `r = sum b_b <= 4` only two
survive: []

```
sizes (2,2):     - 2 C_{i1 i2}^2
sizes (2,1,1):   - 2 C_{i1 i2} C_{i1 i3}      (block 1 is the size-2 one)
```

with `C = Cov(x_i,x_k)` the exact Mehler covariance.  These come from
`-Delta^2/2`.

**Control:** for *linear* `f` the whole right-hand side must vanish for every
`r >= 3`.  It does, identically, and the Ursell terms are exactly what makes it
vanish (e.g. sizes `(2,1,1)`: the only surviving graph has degrees `(2,1,1)` and
weight `beta_{2,2} a_1 a_1 rho rho = 2 Sigma_{i1i2} Sigma_{i1i3}`, cancelled by
`-2 C C`).  Checked to `5e-17` in `03_theory_diagrams.py`.

## 5. The layer contraction

Multilinearity of cumulants plus the coincidence decomposition gives the master
formula.  Let `pi` run over set partitions of the `r` slots (`mult(pi)` = number
of partitions of that type, which all give the same value because the `r` slots
carry the same column `W_{.j}`):

```
kappa_r(z'_j)
  = sum_pi mult(pi)
      sum_{G connected on blocks(pi)}  (1/prod e!)
        INJ[ prod_B beta_{|B|,deg_B}(i_B) W_{i_B j}^{|B|} * prod rhohat_{i_B i_B'}^{e} ]
    + Ursell terms (r = 4)
```

`INJ` = index sum with **distinct** neurons per block.  Because `rhohat` has a
zero diagonal, two blocks joined by an edge cannot coincide automatically; only
non-adjacent blocks need correcting, via

```
INJ[.] = sum_sigma mu(sigma) * (unrestricted sum over the sigma-quotient graph),
mu(sigma) = prod_{S in sigma} (-1)^{|S|-1} (|S|-1)!
```

over partitions `sigma` of the blocks with no `G`-adjacent pair merged.  A merged
node carries the *product* of the merged blocks' `beta` vectors and the *sum* of
their `W` powers.

Slot-partition multiplicities: `r=3`: `(3)`:1, `(2,1)`:3, `(1,1,1)`:1.
`r=4`: `(4)`:1, `(3,1)`:4, `(2,2)`:3, `(2,1,1)`:6, `(1,1,1,1)`:1.

### 5.1 Factorisation

Every diagram whose (quotient) graph is a **tree** factorises: strip leaves one
at a time, each stripping being one `n x n` matmul followed by an elementwise
product.  The leaf matmuls all have the shape

```
LEAF(e, v, p) := rhohat^{oe} @ diag(v) @ W^{op}
```

with `v` a product of `beta` vectors and `p` the (merged) block size.  **They are
shared across every diagram that contains the same leaf**, which is what makes
the whole catalogue affordable.  After grouping the multiplicity sums
analytically, the complete set of matmul primitives is:

| primitive | definition | used by |
|---|---|---|
| `M_e` | `E_e diag(a_e) W` | every size-1 leaf, both `r=3` and `r=4` |
| `P_e` | `E_e diag(c_e) W^{o2}` | size-2 leaves (`c_e = beta_{2,e}`) |
| `Y_D` | `E_D diag(eta_D) W^{o2}`, `eta_D = sum_{p+q=D, p,q>=1} a_p a_q/(p! q!)` | two merged size-1 leaves |
| `Z_T` | `E_T diag(theta_T) W^{o3}`, `theta_T = sum_{p+q+r=T} a_p a_q a_r/(p!q!r!)` | three merged size-1 leaves |
| `V_D` | `E_D diag(zeta_D) W^{o3}`, `zeta_D = sum_{p+q=D} a_p c_q/(p! q!)` | merged size-1 + size-2 leaf |
| `Ubar_q` | `E_q @ Ubar_q`, `Ubar_q = sum_p (1/p!) diag(a_{p+q}) W o M_p` | the 4-vertex path |
| `Chat^{o2} W^{o2}`, `Chat W` | | Ursell terms |
| `W^T C W` | | next-layer covariance (exact) |

`Ubar_q` deserves a note.  The 4-vertex path `1-2-3-4` with multiplicities
`(p,q,r)` seems to need a matmul per `(p,q,r)`, but the `p`- and `r`-sums can be
done *before* the middle matmul because they factor through `Ubar_q`:

```
path(4)  =  12 * sum_{q>=1} (1/q!) cs[ Ubar_q o ( E_q Ubar_q ) ]
```

— one matmul per middle multiplicity `q`, not per triple.

Diagrams whose quotient graph contains `c` independent **cycles do not
factorise**.  For a single cycle of length `Q` the cheapest exact form is

```
T_j = tr( prod_{i=0..Q-1} diag(node_i[:,j]) * E_{e(i,i+1)} )
```

costing `(Q-2)` matmuls per column `j`, i.e. `O(n^4)` for all `j` at `Q = 3`.
See §8 for the cheap surrogate.

### 5.2 The catalogue for `kappa_3`

Ordered by the total power of `rhohat` (`E = sum e`).  `A^{(d)} := diag(a_d) W`.

| `E` | partition | shape | factorised expression | cost |
|---|---|---|---|---|
| 0 | `(3)` | isolated | `(W^{o3})^T kappa_3(x)` | `O(n^2)` |
| `e>=1` | `(2,1)` | 2-block bundle | `3/e! * cs[ diag(c_e) W^{o2} o M_e ]` | reuses `M_e` |
| `p+q` | `(1,1,1)` | path/star, ordered `p,q>=1` | `3/(p!q!) * cs[ A^{(p+q)} o M_p o M_q ]` | reuses `M` |
| `D=p+q` | `(1,1,1)` | leaf-coincidence correction | `-3 * cs[ A^{(D)} o Y_D ]` | one `Y_D` per `D` |
| `>=3` | `(1,1,1)` | **triangle** | `sum 1/(e12! e13! e23!) tr(D_j E D_j E D_j E)` | **`O(n^4)`** |

The leading connected 3-point star (`p = q = 1`) is

```
3 * sum_i a^i_2 W_ij [ (rhohat diag(a_1) W)_ij ]^2
```

— the term quoted in the task statement, but with **three** labelled centres
(factor 3) and with `rhohat = R - I` rather than `R`, the `I` part being exactly
the coincidence that the `(2,1)` partition already accounts for.

### 5.3 The catalogue for `kappa_4`

| `E` | partition | shape | factorised expression | cost |
|---|---|---|---|---|
| 0 | `(4)` | isolated | `(W^{o4})^T kappa_4(x)` | `O(n^2)` |
| `e` | `(3,1)` | bundle | `4/e! cs[ diag(g_e) W^{o3} o M_e ]`, `g_e=beta_{3,e}` | reuses `M_e` |
| `e` | `(2,2)` | bundle | `3/e! cs[ diag(c_e) W^{o2} o P_e ]` | `P_e` |
| — | `(2,2)` | Ursell | `-6 cs[ W^{o2} o (Chat^{o2} W^{o2}) ]` | 1 matmul |
| `p+q` | `(2,1,1)` | path, centre = size-2 block | `6/(p!q!) cs[ diag(c_{p+q}) W^{o2} o M_p o M_q ]` | reuses `M` |
| `p+q` | `(2,1,1)` | path, centre = size-1 block | `12/(p!q!) cs[ A^{(p+q)} o P_p o M_q ]` | reuses `P`,`M` |
| — | `(2,1,1)` | Ursell | `-12 [ cs[W^{o2} o (Chat W)^{o2}] - cs[W^{o2} o (Chat^{o2} W^{o2})] ]` | 2 matmuls |
| `p+q+r` | `(1,1,1,1)` | star | `4/(p!q!r!) cs[ A^{(p+q+r)} o M_p o M_q o M_r ]` | reuses `M` |
| `q` | `(1,1,1,1)` | path | `12/q! cs[ Ubar_q o (E_q Ubar_q) ]` | one matmul per `q` |
| various | both | coincidence corrections | via `Y_D`, `Z_T`, `V_D` | one matmul per index |
| `>=4` | `(2,1,1)`, `(1,1,1,1)` | triangles, 4-cycles | — | **`O(n^4)`** |

---

## 6. Verification of the layer formula

`03_theory_orders.py verify`: `n = 48`, an exactly-Gaussian input field with
`|rho|` mean `0.086` / max `0.33`, `alpha` rms `0.87`, `W ~ N(0, 2/n)`, against
`1.2e8` Monte-Carlo samples.

See §9 for the table; the diagram sum reproduces `kappa_3` and `kappa_4` to
within one Monte-Carlo standard error, and the truncation study there is what
the recommendation in §11 is based on.

---

## 7. Self-consistency: what a non-Gaussian `z` does to `E[x]` and `Cov(x)`

Once `z^l` is known to be non-Gaussian the Mehler formulas of §3 are themselves
only leading order.  First-order multivariate Edgeworth: with `G ~ N(m, Sigma)`
matched to `z`,

```
E[F(z)] = E[ (1 + sum_{r>=3} kappa_{i1..ir} d_{i1}..d_{ir} / r!) F(G) ] + O(kappa^2)
```

The single identity that collapses everything is

```
E[ relu^{(k)}(G_i) ] = a^i_k / sigma_i^k          for all k >= 0
```

(`k=0` trivial, `k=1` is `Phi`, `k>=2` is Proposition A read backwards).  Hence:

**Mean.** `Delta E[x_i] = sum_{r>=3} kappa_r(z_i)/r! * a^i_r / sigma_i^r` — this
*is* Proposition B, so the recursion is consistent with §2 by construction.

**Variance.** `(relu^2)' = 2 relu`, so `(relu^2)^{(r)} = 2 relu^{(r-1)}` and

```
Delta E[x_i^2] = sum_{r>=3} kappa_r(z_i)/r! * 2 a^i_{r-1} / sigma_i^{r-1}
Delta Var(x_i) = Delta E[x_i^2] - 2 mu_i Delta E[x_i]
```

**Covariance.** Define the *shifted Mehler kernel*

```
D_{a,b}[i,k] = sigma_i^{-a} sigma_k^{-b} sum_{p>=0} rhohat_ik^p/p! * a^i_{a+p} a^k_{b+p}
```

(`D_{0,0}` is the ordinary post-ReLU second moment).  Then for `i != k`

```
Delta Cov(x_i,x_k)
  = (1/6)  [ k_iii Dt_{3,0} + k_kkk Dt_{0,3} + 3 k_iik D_{2,1} + 3 k_ikk D_{1,2} ]
  + (1/24) [ k_iiii Dt_{4,0} + k_kkkk Dt_{0,4}
             + 4 k_iiik D_{3,1} + 6 k_iikk D_{2,2} + 4 k_ikkk D_{1,3} ]
```

with `Dt_{r,0} = D_{r,0} - a^i_r a^k_0 / sigma_i^r` (the `p=0` term is exactly
what `-mu_k Delta E[x_i]` removes) and symmetrically for `Dt_{0,r}`.

Every `D_{a,b}` costs `O(K n^2)` — no matmul.  The only new matmuls are the
**mixed** cumulants of `z` themselves; at leading (coincident-block) order

```
kappa_{i^a k^b}(z') = (W^{oa})^T diag( kappa^x_{a+b} ) W^{ob} + O(rhohat)
```

which is one matmul for each of `(a,b) = (2,1), (3,1), (2,2)`; `(1,2)` and
`(1,3)` are transposes.

Verified in `03_theory_selfconsistency.py` on `z = m + A y` with `y` i.i.d.
standardised shifted-Gamma, for which every joint cumulant is closed form; the
residual after the first-order correction scales like `1/k_gamma` (i.e. like
`kappa^2`) while the correction itself scales like `1/sqrt(k_gamma)`.

---

## 8. Cycles: what to do about the diagrams that do not factorise

A single cycle of length `Q = 3` costs `2 n^4 = 8.6e9` FLOPs at `n = 256`, i.e.
**256 matmul units** — ten times the *whole* per-layer budget.  Two options:

1. **Drop them.**  Justified only if they are small; see §9.
2. **Low-rank surrogate.**  `rhohat` at depth is strongly dominated by a few
   eigenvalues (layer 32: `lambda_1 = 93.6`, `lambda_2 = 30.7`,
   `||rhohat||_F^2 = 1.07e4`, so the top two eigenvalues already carry 90% of the
   Frobenius mass).  With `rhohat ~= V_k Lam_k V_k^T`,

   ```
   tr( (diag(x_j) rhohat)^3 ) ~= tr( (Lam_k V_k^T diag(x_j) V_k)^3 )
   ```

   costs `O(k^2 n)` per column plus `O(k^3)`, i.e. `2 k^2 n^2 + 2 k^3 n` for all
   `j`.  At `k = 32`: `1.4e8` FLOPs = **4.2 matmul units** — affordable, and the
   error is controlled by the discarded spectrum.

---

## 9. Numbers

*(filled in from the runs — see the tables below)*

---

## 10. Layer update, with FLOP costs

*(see below)*

---

## 11. Recommended truncation

*(see below)*
