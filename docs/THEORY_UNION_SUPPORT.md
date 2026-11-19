# Helly-type structure and pairwise interleaving of common support

This note states and proves the structural results on common support that
go into the manuscript's Theory section. Notation follows
`docs/MATHEMATICAL_SPEC.md` and the solver in `src/cspph`.

## Setting

* An atom `v` is a ball center `x_v ∈ R^3` with van der Waals radius `r_v`.
  At scale `t ≥ 0` it is enlarged to `B_v(t) = B(x_v, r_v + t)`.
* For a finite atom set `V`,
  `τ(V) = max(0, min_x max_{v∈V} (‖x − x_v‖ − r_v))`,
  i.e. the least `t ≥ 0` with `⋂_{v∈V} B_v(t) ≠ ∅` (weighted 1-center,
  solved exactly by the active-set solver).
* A microcontact `c = (a_c, b_c)` is a directed donor–acceptor atom pair;
  its lens at scale `t` is `L_c(t) = B_{a_c}(t) ∩ B_{b_c}(t)`.
* An interaction patch `P` is a finite set of alternative microcontacts;
  its support set is the union `S_P(t) = ⋃_{c∈P} L_c(t)` (non-convex).
* Common support of patches `P_1,…,P_k`:
  `τ(P_1,…,P_k) = min{ t ≥ 0 : ⋂_i S_{P_i}(t) ≠ ∅ }`.
* Pairwise support: `τ_pw = max_{i<j} τ(P_i, P_j)`; for `m ≤ k`,
  `τ^{(m)} = max_{|J| ≤ m} τ(P_J)`. Increment `η = τ − τ_pw`.
* Patch spread: `δ_P = max_{c,c'∈P} max(‖x_{a_c} − x_{a_{c'}}‖, ‖x_{b_c} − x_{b_{c'}}‖)`
  (alternatives share donor and acceptor site, so `δ_P` is a site
  diameter; `δ_P = 0` for a single-contact patch). `δ = max_i δ_{P_i}`.
* `r_min`, `r_max`: smallest and largest radius among the atoms involved.

*Relation to Čech and Rips.* `τ(P_1,…,P_k)` is the birth of the simplex
`{P_1,…,P_k}` in the nerve of the thickened family `{S_{P_i}(t)}`, i.e. a
Čech-type birth; `τ_pw` is the birth in the flag (Rips-type) complex of
the same nerve. It is not, in general, the minimum-enclosing-ball radius of
the patch atoms: `S_P(t)` is a union of lenses, so only after a selection
`σ` (Lemma 1) does one get a weighted 1-center problem. With zero radii
and one point per patch, `τ` is exactly the ordinary Čech filtration value
(minimum-enclosing-ball radius) of the points. Because the `S_P(t)` are
non-convex, no good-cover nerve theorem is claimed: the resulting barcodes
are not asserted to describe the union of support sets.

## Lemma 1 (selection formula)

`τ(P_1,…,P_k) = min_{σ} τ(V_σ)`, where `σ` ranges over selections
`σ(i) ∈ P_i` and `V_σ = ⋃_i {a_{σ(i)}, b_{σ(i)}}`.

*Proof.* Intersection distributes over union:
`⋂_i ⋃_{c∈P_i} L_c(t) = ⋃_σ ⋂_i L_{σ(i)}(t) = ⋃_σ ⋂_{v∈V_σ} B_v(t)`.
The left side is nonempty iff some term on the right is. ∎

This is exactly what `compute_patch_support` evaluates (branch-and-bound
over selections, exact ball solve per surviving selection).

## Theorem 1 (Helly exactness for single-contact patches; selection–Helly for unions)

(a) For any atom set `V ⊂ R^3`, `τ(V) = max_{W⊆V, |W|≤4} τ(W)`.

(b) If every patch has one microcontact, `τ(P_1,…,P_k) = τ^{(4)}`:
common support is determined by sub-families of at most four patches.

(c) In general, `τ(P_1,…,P_k) = min_σ max_{|J|≤4} τ(V_{σ|J}) ≥ τ^{(4)} ≥ τ^{(3)} ≥ τ_pw`,
and the first inequality (the min–max exchange) can be strict.

*Proof.* (a) "≥" is monotonicity. Let `t* = max_{|W|≤4} τ(W)`. At `t*`
every four of the convex compact sets `B_v(t*)` meet, so by Helly's theorem
in `R^3` all of them meet; hence `τ(V) ≤ t*`. (b) Four atoms belong to at
most four contacts, hence to at most four patches, so (a) gives
`τ(V) ≤ max_{|J|≤4} τ(P_J)`; the reverse is monotonicity. (c) Apply (a) to
each `V_σ` in Lemma 1 and use `min_σ max_J ≥ max_J min_σ`. Strictness:
Proposition 2. ∎

Part (a) is the mathematical content of the solver's `≤ 4` active-ball
certificate; (b)–(c) say that the only source of order-`>4` information is
the choice among alternatives. Order-3 and order-4 information
(`τ > τ_pw`) exists already for single-contact patches.

## Proposition 2 (no fixed order suffices for union patches)

For every `k ≥ 3` and every `D > 2r` there are `k` patches, each a union of
`k − 1` lenses with radius `r`, such that every `(k−1)`-subfamily has
support `0` while `τ(P_1,…,P_k) = D/2 − r`. In particular `τ_pw = 0`,
`η = D/2 − r` is arbitrarily large, and no inequality
`τ ≤ f(τ_pw)` or `τ ≤ τ^{(m)}` with `m < k` holds for union patches.

*Construction.* Take points `p_1,…,p_k` with all pairwise distances `≥ D`
(e.g. equilateral triangle for `k = 3`, regular tetrahedron for `k = 4`,
side `D`). Let `P_i` contain one contact at each `p_j, j ≠ i` (donor and
acceptor at, or arbitrarily close to, `p_j`).

*Proof.* A subfamily missing `P_j` meets at `p_j` at `t = 0`. For the full
family, a selection cannot use every point: `P_j` has no contact at `p_j`,
so the selected atoms sit at at least two distinct points, and the support
of two balls of radius `r` at distance `≥ D` is `≥ D/2 − r`; selecting
`p_2` for `P_1` and `p_1` for all others attains `D/2 − r` for the equal-side
configurations above. ∎

*Minimal case* (`k = 3`, two alternatives per patch, a "hollow triangle"):
every pair of patches shares a point, the triple does not; the nerve of the
three supports is the boundary of a triangle. This is the smallest
configuration in which pairwise compatibility and common support disagree
with `τ_pw = 0`.

*Remark.* With unrestricted radii, two alternatives per patch already admit
no finite Helly number: on a line, `S_i = [½, i − ½] ∪ [i + ½, k + ½]`
(each interval is the trace of a ball centered on the line, and balls
centered on a common line meet iff their traces do) gives `k` sets, any
`k − 1` of which meet, with empty total intersection. With van der Waals
radii from a fixed table the construction of Proposition 2 is the relevant
one.

## Theorem 3 (affine Jung bound)

For any patches,
`τ_pw ≤ τ(P_1,…,P_k) ≤ max(τ_pw, √(3/2) · (r_max + τ_pw + δ) − r_min)`.

Equivalently, in the radius variable: `r_min + τ ≤ √(3/2) (r_max + τ_pw + δ)`.

*Proof.* Lower bound: any selection for the whole family restricts to a
selection for each pair (Lemma 1). Upper bound: fix any selection `σ`. For
each pair `i ≠ j` there are `c ∈ P_i`, `c' ∈ P_j` and a witness
`x_{ij} ∈ L_c(τ_pw) ∩ L_{c'}(τ_pw)`. For an atom `u` of `σ(i)` with the same
role (donor/acceptor) as atom `u_c` of `c`,
`‖x_u − x_{ij}‖ ≤ ‖x_u − x_{u_c}‖ + ‖x_{u_c} − x_{ij}‖ ≤ δ + r_max + τ_pw`.
The same bound holds for atoms of `σ(j)` (via `c'`) and, using any `j ≠ i`,
for both atoms of `σ(i)` against each other. Hence the `2k` centers of
`V_σ` have diameter at most `2(r_max + τ_pw + δ)`. By Jung's theorem in
`R^3` they lie in a ball of radius `√(3/8) · diam`, whose center `z`
satisfies `‖z − x_v‖ ≤ √(3/2)(r_max + τ_pw + δ)` for every `v ∈ V_σ`.
So `z ∈ B_v(t)` for all `v` once `r_min + t ≥ √(3/2)(r_max + τ_pw + δ)`. ∎

*Tightness (convex case).* Four single-atom-site contacts at the vertices
of a regular tetrahedron of side `2r`, equal radii `r`, `δ = 0`: pairwise
balls touch, so `τ_pw = 0`, and `τ = √(3/8)·2r − r = (√(3/2) − 1) r`,
equality in Theorem 3. For `r = 1.6 Å`, `η = 0.36 Å` with `τ_pw = 0`: an
increment is created at the very first scale even for convex patches.

*Necessity of δ.* In Proposition 2, `δ = D` and `η = D/2 − r`, so the
increment must be allowed to grow linearly in the patch spread; Theorem 3
gives coefficient `√(3/2)`, Proposition 2 shows at least `1/2`.

*Reading.* Pairwise support does not recover common support even for
single-contact patches: the tetrahedron already has `τ_pw = 0 < τ`, and
`η > 0` is expected on real single-contact paths. For such patches,
however, the increment is a bounded affine perturbation of pairwise support
(Jung constant √(3/2)) and is decided by quadruples (Theorem 1b). The strict
excess of common support over the quadruple value `τ^(4)`, and any increment
that grows with geometry rather than with `r_max − r_min`, comes only from
union-valued alternatives and is controlled by their spread `δ`.

## Theorem 4 (interleaving of persistent path modules)

Fix a directed patch graph. For a birth rule `β`, let `A^β_t` be the set of
allowed elementary paths whose patch set has `β ≤ t`, with
`β ∈ {τ, τ_pw}` evaluated on the patch set of the path. Both rules are
monotone under taking sub-paths, so each `A^β_t` is a path complex,
inclusions `A^β_s ⊆ A^β_t` induce chain maps on the GLMY `Ω_•`, and
`H_n(Ω_•(A^β_t))` is a persistence module `M^β_n`.

Let `φ(t) = max(t, √(3/2)(r_max + t + δ) − r_min)` with `δ, r_min, r_max`
taken over all patches of the graph. Then
`A^τ_t ⊆ A^{τ_pw}_t ⊆ A^τ_{φ(t)}` for all `t ≥ 0`, and

1. on the logarithmic scale `s = log(t + r_min)` the modules `M^τ_n` and
   `M^{τ_pw}_n` are `ε`-interleaved with
   `ε = ½ log(3/2) + log((r_max + δ)/r_min)`, hence
   `d_B(Bar^τ_n, Bar^{τ_pw}_n) ≤ ε` for the log-scale barcodes;
2. on the original scale restricted to `[0, T]` (`T` the largest finite
   birth), `d_B ≤ (√(3/2) − 1) T + √(3/2)(r_max + δ) − r_min`.

*Proof.* The inclusions are Theorem 3 applied path by path (the first is
the lower bound, the second the upper bound). For (1): with
`g(t) = log(t + r_min)`,
`g(φ(t)) − g(t) ≤ log(√(3/2)(t + r_max + δ)/(t + r_min)) ≤ log(√(3/2)(r_max + δ)/r_min)`
because `r_max + δ ≥ r_min`. Two nested filtrations with
`F_t ⊆ G_t ⊆ F_{t+ε}` (here in `g`-coordinates) give an `ε`-interleaving
of the induced modules, and the algebraic stability (isometry) theorem
bounds the bottleneck distance by `ε`; the modules are pointwise finite
dimensional because the complexes are finite. (2) is the same argument
with `φ(t) − t ≤ (√(3/2) − 1)T + √(3/2)(r_max + δ) − r_min` on `[0, T]`. ∎

*Values.* For donor/acceptor atoms N and O (`r = 1.55, 1.52 Å`) and
single-contact patches, `ε = 0.22` (a factor `1.25` in `t + r_min`). With
site spreads typical of carboxylate or phosphate oxygens (`δ ≈ 2.2–2.5 Å`)
the bound is no longer small: the theory predicts that barcode
divergence between the two rules is driven by patch spread, which is
testable on the existing deep subset (see below).

## What is and is not claimed

* Proven: Lemma 1, Theorem 1, Proposition 2, Theorem 3 (with the tight
  tetrahedron), Theorem 4.
* Not claimed: a sharp constant in the union case (between `1/2` and
  `√(3/2)` in front of `δ`); a bounded Helly number for union patches with a
  fixed number of alternatives and van der Waals radii (open; the interval
  remark shows radii matter).
* The earlier statement "common support is determined by at most four
  patches" is true only for single-contact patches (Theorem 1b); for union
  patches it is false (Proposition 2) and should be stated as the
  selection–Helly formula (Theorem 1c).

## Suggested numerical checks on real structures

1. Per path: ratio `(r_min + τ)/(r_max + τ_pw + δ)` ≤ `√(3/2)` (Theorem 3),
   its distribution, and the fraction of paths with `τ > τ^{(4)}`
   (min–max gap, Theorem 1c).
2. Correlation of per-path `η` with patch spread `δ`, and of per-structure
   barcode divergence with mean `δ` (RNA vs protein–DNA).
3. Synthetic checks of Theorems 1–3 and Proposition 2 with the repository
   solver: `experiments/theory_checks/`.

## References

* E. Helly (1923); H. Jung, *J. reine angew. Math.* 123 (1901).
* F. Chazal, V. de Silva, S. Oudot, Persistence stability for geometric
  complexes, *Geom. Dedicata* 173 (2014) — Čech/Rips interleaving via Jung.
* F. Chazal, D. Cohen-Steiner, M. Glisse, L. Guibas, S. Oudot, Proximity of
  persistence modules and their diagrams, SoCG 2009; U. Bauer, M. Lesnick,
  Induced matchings and the algebraic stability of persistence barcodes,
  *J. Comput. Geom.* 6 (2015).
* A. Grigor'yan, Y. Lin, Y. Muranov, S.-T. Yau, Homologies of path complexes
  and digraphs (2012); S. Chowdhury, F. Mémoli, Persistent path homology of
  directed networks, SODA 2018.
* N. Amenta, A short proof of an interesting Helly-type theorem, *DCG* 15
  (1996); J. Matoušek, A Helly-type theorem for unions of convex sets,
  *DCG* 18 (1997); G. Kalai, R. Meshulam, Leray numbers of projections and
  a topological Helly-type theorem, *J. Topology* 1 (2008).
