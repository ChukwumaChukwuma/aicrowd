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
| full recursion vs Monte Carlo, transport, oracle ceiling | `scripts/03_theory_scheme.py` |

Modes: `03_theory_orders.py {verify,orders,factorise,cost}`,
`03_theory_scheme.py {,"" transport, oracle, cost}`.

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
3 * cs[ A^(2) o M_1 o M_1 ]  -  3 * cs[ A^(2) o Y_2 ]
  = 3 sum_i a^i_2 W_ij [ (rhohat diag(a_1) W)_ij ]^2
    - 3 sum_i a^i_2 W_ij ( rhohat^{o2} diag(a_1 o a_1) W^{o2} )_ij
```

i.e. the term quoted in the task statement, with three corrections:

* a factor **3** — a path on three labelled slots has three choices of centre;
* `rhohat = R - I` rather than `R`.  The `I` part of `R` is exactly the
  index coincidence that the `(2,1)` slot-partition already accounts for with
  its own (resummed, and therefore different) block weight `c_e`, so using `R`
  double-counts it with the wrong coefficient;
* the second line, which enforces that the two *leaves* carry distinct neurons.
  It is **76% of this diagram** at `n = 48` (`03_theory_orders.py factorise`),
  not a small correction.

Both closed forms are pinned against the generic diagram engine to `1.5e-17` by
`03_theory_orders.py factorise`.

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

### 9.1 What the field actually looks like (`n = 256`, seed 0)

Gaussian covariance propagation, `scripts/03_theory_orders.py orders`:

| layer | `|rho|` mean | `rho` rms | `sum_k rho_ik^2` | `alpha` rms | `lam_1(rhohat)` |
|---|---|---|---|---|---|
| 1 | 0.050 | 0.063 | 1.00 | 0.00 | 3.8 |
| 4 | 0.091 | 0.114 | 3.3 | 1.25 | 10.9 |
| 8 | 0.136 | 0.169 | 7.3 | 2.08 | 24.8 |
| 16 | 0.180 | 0.223 | 12.6 | 2.65 | 37.8 |
| 24 | 0.256 | 0.313 | 25.0 | 3.84 | 69.3 |
| 32 | 0.336 | 0.404 | 41.6 | 4.49 | 93.6 |

Two facts set everything:

* **`rhohat` is not small at depth.**  The naive "`rho ~ n^{-1/2}`, so diagram
  order = order in `1/sqrt(n)`" counting is valid only in the first few layers.
  By layer 32 the mean `|rho|` is 0.34 and the max is 0.94, so the expansion
  parameter is `O(1/3)`, not `O(1/16)`, and orders must be counted empirically.
* **`alpha` is large.**  Only ~24% of neurons have `|alpha| < 2` at layer 32, and
  `sigma phi(alpha)` — the prefactor of every non-Gaussian correction to
  `E[relu]` — has rms `3.4e-2` against a mean activation of ~1.9.  Corrections
  are carried by a minority of near-kink neurons.

And the true non-Gaussianity, from `1.2e6`–`3e6` Monte-Carlo samples
(two independent seeds agree to three digits):

| layer | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| `gamma_3` rms | 0.002 | 0.040 | 0.094 | 0.203 | 0.326 | 0.412 | 0.442 |
| `gamma_4` rms | 0.005 | 0.041 | 0.095 | 0.180 | 0.334 | 0.415 | 0.427 |

### 9.2 The layer formula, against Monte Carlo

`03_theory_orders.py verify`: `n = 48`, exactly-Gaussian input field
(`|rho|` mean 0.086 / max 0.33, `alpha` rms 0.87), `N = 1.2e8`.

```
mean of z:  rms err 7.9e-05   (mc se 7.9e-05)     <- exact identity, m' = W^T mu
var  of z:  rms err 8.5e-05                       <- exact identity, S' = W^T C W
kappa_3:    rms |diagram - mc| = 1.92e-04   mc se 1.5e-04   rms kappa_3 = 7.55e-02
kappa_4:    rms |diagram - mc| = 2.93e-04   mc se 2.7e-04   rms kappa_4 = 4.39e-02
```

Both cumulants are reproduced at the Monte-Carlo noise floor — 0.25% and 0.7% of
their own size.

### 9.3 Order counting: which diagrams matter

Fractions of the total rms at `n = 256` (`orders` mode; the "order" column is
the total power of `rhohat`, "ursell" terms are labelled order 0):

**`kappa_3`** — well behaved, three comparable contributions:

| diagram | layer 3 | layer 9 | layer 17 | layer 25 |
|---|---|---|---|---|
| `(1,1,1)` path, order 2 (**the star**) | 0.32 | 0.65 | 0.66 | 0.64 |
| `(2,1)` bundle, order 1 | 0.58 | 0.51 | 0.56 | 0.42 |
| `(2,1)` bundle, order 2 | 0.27 | 0.49 | 0.44 | 0.48 |
| `(3)` all-coincident | 0.52 | 0.18 | 0.12 | 0.10 |
| `(1,1,1)` path, order 3 | 0.010 | 0.033 | 0.033 | 0.030 |
| `(1,1,1)` **triangle**, order 3 | 0.005 | 0.006 | 0.007 | 0.005 |

**`kappa_4`** — dominated by a *cancelling pair*.  The `(2,1,1)` order-2 tree and
its Ursell partner `-2 C_ik C_il` are each 100x the total and cancel to four
digits; likewise `(2,2)` order-2 against `-2 C_ik^2`:

| diagram | layer 3 | layer 9 | layer 17 | layer 25 |
|---|---|---|---|---|
| `(2,1,1)` tree order 2 | 0.52 | 12.5 | 13.8 | 112.9 |
| `(2,1,1)` ursell | 0.45 | 12.6 | 14.0 | 113.1 |
| `(2,2)` tree order 2 | 1.35 | 9.3 | 10.7 | 53.6 |
| `(2,2)` ursell | 0.98 | 9.2 | 11.0 | 53.8 |
| `(2,1,1)` order 3 | 0.06 | 0.63 | 0.72 | 0.72 |
| `(1,1,1,1)` order 3 (**star + path**) | 0.09 | 0.54 | 0.48 | 0.53 |
| `(1,1,1,1)`/`(2,1,1)` cycles, order 3-4 | 0.02 | 0.15 | 0.16 | 0.11 |

> **Numerical warning.**  `kappa_4` is a difference of two quantities 100x its
> own size.  The Ursell term and its graph partner must be summed *in the same
> expression*, and `float32` (relative `6e-8`, so `6e-6` after a 100x
> cancellation) is marginal.  Do the `kappa_4` accumulation in float64.

Truncation cost, measured as the rms deviation from the full sum:

| truncation | `kappa_3` (L3 / L9 / L17 / L25) | `kappa_4` (L3 / L9 / L17 / L25) |
|---|---|---|
| drop **all cycles** | 0.5% / 0.6% / 0.7% / 0.5% | 0.4% / 1.6% / 1.5% / 1.0% |
| `emax` 8/4/3 | 0.0% / 0.1% / 0.1% / 0.1% | 0.6% / 4.9% / 6.4% / 4.7% |
| `emax` 6/3/2 | 0.1% / 0.3% / 0.4% / 0.4% | 8.8% / 52% / 45% / 52% |
| `emax` 4/2/2 | 1.1% / 3.3% / 3.2% / 3.0% | 11% / 81% / 71% / 85% |
| **no injectivity correction** | 23% / 48% / 45% / — | 36% / 71% / 89% / — |
| coincident blocks only | 32% / 65% / 65% / — | 51% / >100% / >100% / — |

Read off:

1. **Cycles can be dropped** at the 1% level.  Good, because they are the only
   diagrams that do not factorise.
2. **The injectivity (index-coincidence) corrections cannot be dropped.**  They
   are 25–50% of `kappa_3`.  The reason is structural: `W_ij^2` has mean `2/n`
   while `W_ij W_kj` (`i != k`) has mean zero, so the `n` coincident terms are
   coherent and match the `n^2` incoherent ones in size.  Any treatment that
   uses `R` instead of `rhohat = R - I` and forgets the diagonal is wrong at the
   tens-of-percent level.
3. **`kappa_4` needs a higher edge order than `kappa_3`** — `emax` 8/4/3 leaves
   5%, and the reason is the cancellation above: what survives is a small
   residue of large terms, so relative accuracy in the individual terms
   translates to a much larger relative error in the total.  The 2-block sums
   should be run to `K_2 = 16`, which the edge-kernel folding (§10) makes free.

### 9.4 Where the final-layer error actually lives

Error-injection: restart the Gaussian recursion from the Monte-Carlo-exact
`(mu^l, C^l)` at layer `l` and measure the final-layer rms error.

| restart at layer | 1 | 5 | 9 | 13 | 17 | 21 | 25 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|---|
| final rms | 1.23e-2 | 8.1e-3 | 7.3e-3 | 5.7e-3 | 4.9e-3 | 4.3e-3 | 3.6e-3 | 2.0e-3 | 1.33e-3 |

* Error is generated roughly uniformly across depth and accumulates by ~9x.
* **One layer** of the Gaussian assumption costs `1.33e-3` rms.
* With the exact marginal `kappa_3` of `z^32`: `4.8e-4`.  With `kappa_3` and
  `kappa_4`: **`1.5e-4`**.
* At layer 32 the Gaussian scheme's `sigma` is wrong by rms `8.1e-2` (21%
  relative) and its `m` by `1.7e-2`.  The final error `1.24e-2` decomposes as
  `Phi * dm + phi * ds` = `1.231e-2` + `7.9e-4`: it is carried almost entirely
  by the error in the **mean** `m^32 = W^T mu^31`.  (`ds` is large but the
  neurons whose `sigma` is most wrong are the ones with large `|alpha|`, where
  `phi(alpha)` is tiny; `sigma` errors hurt indirectly, by corrupting
  `mu = m Phi + sigma phi` at every earlier layer and thus `m` downstream.)
  **The error is in the propagated moments, not in the last rectification.**

### 9.5 The Gram-Charlier wall

Accumulating `c_r = E[He_r((z^32 - m)/sigma)]` from `3e6` samples and summing
`E[relu(z)] = sum_r c_r a_r / r!` term by term:

| `R` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| rms err | 1.33e-3 | 4.83e-4 | **1.47e-4** | 1.96e-4 | 1.84e-4 | 1.18e-4 | 1.39e-4 | 2.13e-4 | 4.11e-4 |
| `c_R` rms | 0.001 | 0.44 | 0.43 | 0.50 | 2.97 | 8.74 | 25.5 | 542 | 1.7e4 |

`c_r / sqrt(r!)` does **not** decay (`c_12/sqrt(12!) = 0.79`), so `p_Z/phi` is not
in `L^2(gamma)`: the real `z^32` has heavier tails than the matched Gaussian and
the Gram-Charlier series is *asymptotic, not convergent*.  Its optimal
truncation is `R = 4` — exactly `kappa_3` and `kappa_4` — and it stalls at
`1.5e-4` per layer no matter how many cumulants are added.

A moment-matched shape family does not rescue it: a two-component Gaussian
mixture matched to `(kappa_1..kappa_4)` (feasible for every neuron here, since
`kappa_4 > 0`) gives rms `2.1e-3` — *worse* than Gaussian-plus-Edgeworth,
because the family is bimodal and the truth is not.

**Consequence.**  Even a perfect `kappa_3/kappa_4` scheme cannot do better than
`1.5e-4` per layer.  If the per-layer improvement carries through the ~9x
accumulation, the best attainable final rms is about `1.4e-3`
(MSE `~2e-6`, adjusted score `~2e-7`) against the Gaussian baseline's
`1.24e-2` (MSE `1.53e-4`).  That is an 80x MSE gain, and it is **not** the
`1.8e-10` noise floor.  Reaching `1.3e-5` rms needs the *marginal law*, not more
cumulants of it.

### 9.6 The term the task statement does not contain: cumulant transport

Running the whole recursion (`03_theory_scheme.py`) exposes it.  Feeding the
scheme's own diagram-computed cumulants gives, at layer 32, `kappa_3` **rms
`8.9e-2` too small out of a true `8.97e-2`** — i.e. the diagram sum produces
almost none of the observed skewness (`gamma_3 = 0.045` predicted vs `0.44`
true).

The reason: the diagram catalogue of §5 computes the cumulants of `z^{l+1}`
**assuming `z^l` is Gaussian**.  That is only the *source*.  Applying the same
first-order Edgeworth operator to the joint cumulants of `x` gives, for distinct
neurons,

```
delta kappa(x_i1,x_i2,x_i3) = kappa^{(z)}_{i1 i2 i3} Phi_i1 Phi_i2 Phi_i3 + O(rhohat)
```

(three derivatives on three different factors, `relu' = 1{.}`), so the correct
recursion is a **linear transport plus a source**:

```
kappa^(3)(z^{l+1}) = S^(3)(l) + (Wt^T)^{ox3} kappa^(3)(z^l),   Wt = diag(Phi) W
kappa^(4)(z^{l+1}) = S^(4)(l) + (Wt^T)^{ox4} kappa^(4)(z^l)
```

Obvious in hindsight: a neuron with `|alpha| >> 0` is an affine map and affine
maps carry cumulants through exactly.  With `alpha` rms `4.5` at depth, most of
the network is affine for most inputs, so transport dominates: the observed
`gamma_3 = 0.44` is a geometric accumulation `s/(1-g)` of a per-layer source
`s ~ 0.045` under a transport gain `g ~ 0.9`.

**This breaks the closure.**  Transport mixes *off-diagonal* entries of the
3-index tensor into the diagonal, so `{kappa_3(z_j)}` is not sufficient state;
the full `kappa^(3)_{i1i2i3}` must be carried.  A demonstration
(`03_theory_scheme.py transport`) that carries a rank-growing CP form
`sum_v c_v G_v^{ox3}` with the *coincident-block* source only and transports it
by `Wt` recovers `gamma_3 = 0.065` against a true `0.445` — transport is real
but a diagonal-only source captures only ~15% of it, because 80% of the source
sits in the `(2,1)` and `(1,1,1)` diagrams which are not of cube form.

Confirmation of the consequence: injecting the **exact** marginal `kappa_3` and
`kappa_4` at every layer, while leaving the mixed cumulants
`kappa_iik, kappa_iiik, kappa_iikk` at coincident-block order, moves the final
rms only from `1.237e-2` to `1.143e-2` (8%).  The covariance correction of §7
is only as good as the mixed cumulants fed into it, and those suffer the same
transport deficit.

---

## 10. The layer update, with FLOP costs

### 10.1 Edge-kernel folding — the trick that pays for the high orders

A diagram edge whose multiplicity feeds no other vertex's degree can have its
whole multiplicity sum folded into an elementwise kernel *before* the matmul:

```
sum_{e>=1} (1/e!) sum_{i != k} u^i_e rhohat_ik^e v^k_e W_ij^b W_kj^b'
   = cs[ W^{ob} o ( Xi W^{ob'} ) ],     Xi_ik = sum_e rhohat_ik^e u^i_e v^k_e / e!
```

`Xi` costs `O(K n^2)` and the matmul count is **one, independent of `K`**.  Every
2-block diagram — `(2,1)`, `(3,1)`, `(2,2)`, which are the largest contributors
and the ones needing the highest order because `rhohat` is not small — is of this
form, as is every leaf-coincidence correction whose merged leaf is the only
consumer of the multiplicity.  So `K_2 = 16` is free; only the genuine
3- and 4-vertex trees pay per order.

The 4-vertex path needs the same idea one level up: pre-summing the outer legs
into

```
Ubar_q = sum_p (1/p!) diag(a_{p+q}) W  o  M_p
path(4) = 12 * sum_{q>=1} (1/q!) cs[ Ubar_q o (E_q Ubar_q) ]
```

turns `O(K^3)` matmuls into `K_4 - 2`.

### 10.2 Pseudocode

```
state: m, s, R          (n, n, n x n)      pre-activation moments of z^l
       k3, k4           (n, n)             marginal cumulants of z^l
       K21, K31, K22    (n x n each)       kappa_iik, kappa_iiik, kappa_iikk
       [recommended]    U (n x k), Lam3 (k^3), Lam4 (k^4)   compressed cumulant tensors

per layer, with W = W^{l+1}, rhohat = R - I:

 1  alpha = m/s;  Phi, phi;  a_k (k <= K_2+4);  I_p (p <= 4)          O(K n)
    A^(p)_k, beta_{2,k} = c_k, beta_{3,k} = g_k                       O(K n)
    kappa_2..4 of x                                                   O(n)
 2  mu   = a_0   + [k3/6 * a_3/s^3 + k4/24 * a_4/s^4]                 O(n)
    var  = relu_var + [k3/6 * 2a_2/s^2 + k4/24 * 2a_3/s^3] - 2 mu dmu O(n)
 3  C    = sum_{p=1..K_2} rhohat^{op}/p! * a_p a_p^T                  O(K n^2)
    D_{a,b} shifted-Mehler kernels, (a,b) with a+b <= 4               O(K n^2)
    C   += Delta Cov(x)   [sec 7, uses K21/K31/K22 and D]             O(K n^2)
    fill diagonal of C with var
 4  m'   = W^T mu                                                     O(n^2)
    S'   = W^T C W                                            2 matmuls   [EXACT]
    s'   = sqrt(diag S');  R' = S' / (s' s'^T)                        O(n^2)
 5  kappa_3', kappa_4' from the catalogue of secs 5.2-5.3:
      (3), (4) coincident                                      0 matmuls
      (2,1),(3,1),(2,2) bundles, folded, K_2 = 16              3 matmuls
      M_e, e = 1..K_3-1     (3-vertex path leaves)             K_3-1
      P_e, e = 1..K_4-1     ((2,1,1) paths on a 1-block)       K_4-1
      Ubar chain, q = 1..K_4-2                                 K_4-2
      folded coincidence kernels Y, Z, V                       4 matmuls
      Ursell  Chat^{o2} W^{o2},  Chat W                        2 matmuls
      cycles: DROPPED (0.5-1.6%)
 6  K21', K31', K22' = (W^{oa})^T diag(kappa^x_{a+b}) W^{ob}   3 matmuls
 7  [recommended] transport:  U <- W^T diag(Phi) U ;  re-orthonormalise;
    project the new source onto U; Lam3, Lam4 updated          O(k n^2 + k^4)
```

### 10.3 Budget

One matmul unit = `n^2 (2n-1) = 3.349e7` FLOPs at `n = 256` (the flopscope
convention used by `whestfloor/contract.py`).  Free-compute ceiling
`2.72e10` = **812 matmul units for all 32 layers**, i.e. **25.4 per layer**.

With `K_2 = 16` (free), `K_3 = 4`, `K_4 = 3`:

| primitive | # | FLOPs |
|---|---|---|
| `S' = W^T C W` | 2 | 6.70e7 |
| mixed cumulants `(W^oa)^T diag(k) W^ob` | 3 | 1.00e8 |
| `(2,1)`, `(3,1)`, `(2,2)` bundles (folded) | 3 | 1.00e8 |
| leaf-coincidence kernels `Y`, `Z` (folded) | 2 | 6.70e7 |
| `M_e`, `e = 1..3` | 3 | 1.00e8 |
| `P_e`, `e = 1..2` | 2 | 6.70e7 |
| `Ubar_q` chain | 1 | 3.35e7 |
| `V_D`, `Z_T` corrections | 2 | 6.70e7 |
| Ursell `Chat^{o2} W^{o2}`, `Chat W` | 2 | 6.70e7 |
| elementwise kernels + column sums (~80 passes at `K = 16`) | — | 8.39e7 |
| **per layer** | **20** | **7.54e8** |
| **32 layers** | | **2.41e10** |

`2.41e10 < 2.72e10`: it fits, with ~11% headroom.  What does *not* fit:

| item | FLOPs | matmul units |
|---|---|---|
| one 3-cycle, exact, all `j` (`2 n^4`) | 8.59e9 | 257 |
| same, rank-16 `rhohat` | 3.57e7 | 1.1 |
| same, rank-32 `rhohat` | 1.51e8 | 4.5 |
| exact 3-tensor transport (3 mode products, `6 n^4`) | 2.58e10 | 770 / layer |
| **rank-16 Tucker transport of `kappa^(3)`** | 2.10e6 | **0.06 / layer** |
| **rank-32 Tucker transport of `kappa^(3)`** | 4.19e6 | **0.13 / layer** |

Two things to notice.  The exact cycle and the exact tensor transport are each
10x-1000x the entire budget.  Their **low-rank surrogates are essentially free** —
and `rhohat` is strongly low-rank at depth (layer 32: `lam_1 = 93.6`,
`lam_2 = 30.7`, `||rhohat||_F^2 = 1.07e4`, so the top two eigenvalues already
carry 90% of the Frobenius mass).  Compression is not a compromise here, it is
the natural representation.

---

## 11. Recommended truncation, and what it is worth

### Keep

1. **Everything at edge order `<= 2` for `kappa_3`, `<= 3` for `kappa_4`**, with
   the 2-block bundle sums folded and run to `K_2 = 16`.
2. **All injectivity (index-coincidence) corrections.**  Non-negotiable: 25-50%
   of `kappa_3`.  Use `rhohat = R - I`, never `R`.
3. **Both Ursell terms**, summed in the same expression as their graph partners,
   in float64.  Without them `kappa_4` is wrong by more than 100%.
4. **The self-consistency corrections of §7** to `E[x]`, `Var(x)`, `Cov(x)` —
   they cost no matmuls beyond the three mixed-cumulant products, and they are
   verified to remove 9x-40x of the non-Gaussian error in a controlled setting.

### Drop

5. **All cyclic diagrams** (triangles, 4-cycles, triangle+pendant): 0.5-1.6%,
   and 257 matmul units to do exactly.  If they are ever wanted, use the rank-32
   surrogate at 4.5 units.
6. **Edge orders above 4 in the genuine 3- and 4-vertex trees**: `<0.5%`.
7. **`kappa_5` and beyond**: §9.5 shows they make the answer *worse*.

### The correction the task statement is missing, and which decides the result

8. **Cumulant transport** (§9.6).  Without it the scheme computes ~10% of the
   true skewness at depth and the whole exercise is worth 8%, not 9x.  With it,
   carried as a rank-`k` symmetric Tucker tensor updated by
   `U <- W^T diag(Phi) U`, the cost is 0.06-0.13 matmul units per layer — under
   1% of the budget — leaving the 20-matmul source computation intact.  This is
   where the remaining engineering should go.

### Honest ceiling

Even with transport done perfectly, §9.5 caps a `kappa_3/kappa_4` marginal model
at `1.5e-4` rms per layer, hence ~`1.4e-3` final rms and MSE ~`2e-6`: **80x
better MSE than the Gaussian baseline, still 100x short of the `1.8e-10` noise
floor**.  The obstruction is not the diagram truncation and not the FLOP budget —
it is that the Gram-Charlier series for the true `z^32` diverges.

The route past it is visible in the same numbers.  `rhohat` is dominated by a
handful of modes; the non-Gaussianity is generated by those common modes; and
conditional on them the pre-activation is a sum of ~256 weakly dependent terms
and is far closer to Gaussian than the marginal is.  So the natural next
representation is not "Gaussian + cumulants" but **"Gaussian conditional on `k`
non-Gaussian common factors"**:

```
z^l = m + U xi + eta,   eta | xi ~ Gaussian,   xi in R^k with a quadrature grid
E[relu(z_j)]     = E_xi[ relu_mean( m_j(xi), sigma_j ) ]
Cov(x_i, x_k)    = E_xi[ Mehler_ik(xi) ] + Cov_xi( mu_i(xi), mu_k(xi) )
```

Conditioning a Gaussian on `xi` moves the mean but not the covariance, so the
`Q^k` quadrature nodes share one Mehler structure; the per-node cost is
`Q^k` covariance products, `2 Q^k` matmul units per layer, which fits for
`k = 1, Q <= 8`.  It captures the marginal shape *to all orders* in the dominant
direction, which is exactly what the Gram-Charlier series cannot do.  The
diagram machinery above is not wasted in that scheme — it is what supplies the
conditional cumulants and the residual (non-factor) non-Gaussianity.
