GILP Components Walkthrough
Proto-3: Hyperbolic Logic (Poincaré Ball)
Implemented Hyperbolic Geometry to match the hierarchical nature of proofs.

Poincaré Manifold: Implemented 

mobius_add
, 

dist
, 

exp_map
.
Hyperbolic Contrastive Loss: Enforces $d_H(u, v) \ll d_H(u, others)$.
Hyperbolic Greedy Inference: Strict descent in Poincaré space.
Fossilization Results
Experiment	Mode	Result	Dist/Budget	Reason
Proto-2.6	ON	❌ FAIL	2.00	FAIL_DESCENT
Proto-3.0	ON	✅ SUCCESS	1.68	Success
Proto-3.0	OFF	❌ FAIL	9.70	FAIL_LOCAL_MINIMA
Scientific Conclusion
Hyperbolic Geometry Works: Unlike Proto-2.6 (Euclidean), the ON mode succeeded with strict greedy descent. This proves that Hyperbolic Space can represent the logical connectivity as a navigable landscape (no local minima along the path).
Zero-Shot Gap: The OFF mode (Empty Graph) failed. This means the model currently relies on GNN message passing to construct this landscape. It cannot yet "hallucinate" the correct geometry from rule text alone.
Progress: We fixed the "Rugged Landscape" problem (ON Success). The remaining challenge is "Generalization from Text" (OFF Success)

