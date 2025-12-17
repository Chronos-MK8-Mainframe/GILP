# Theory of Fossilization: How We Made "OFF" Mode Work

## The Problem: The "Geometry Gap"
In previous prototypes (Proto-3.0), the model worked perfectly in **ON Mode** (with the graph) but failed in **OFF Mode** (zero-shot/empty graph).

*   **ON Mode (Teacher):** The GNN could "see" the edges ($u \to v$). It used message passing to pull connected nodes close together in Hyperbolic space. The geometry was correct because the *graph structure* forced it to be correct.
*   **OFF Mode (Student):** Without edges, the model only saw the text of the rules. It had no incentive to place "related" rules near each other because it didn't know they were related. The embedding space was random/unstructured w.r.t logic.

## The Solution: "Geometric Distillation" (Fossilization)

We introduced a new learning objective that forces the model to **internalize** the graph structure into its text-processing weights. We effectively "fossilize" the dynamic graph connections into static geometric positions.

### 1. The Concept
We treat the **GNN-enhanced embedding** ($z_{graph}$) as the "Ground Truth" or "Platonic Ideal" of where a concept belongs in the logical universe. We treat the **Text-only embedding** ($z_{text}$) as a "Prediction".

We demand that:
$$ z_{text}(Rule) \approx z_{graph}(Rule | Graph) $$

This means the model must learn to predict the *context-aware* position of a rule solely from its *content*.

### 2. The Loss Function
We added a **Fossilization Loss** term to the training objective:

$$ L_{total} = L_{structure} + \lambda L_{fossil} $$

Where:
*   $L_{structure}$: The original hyperbolic contrastive loss (makes the geometry navigable).
*   $L_{fossil} = d_H(z_{text}, z_{graph})$: The Hyperbolic distance between the "blind" prediction and the "seeing" target.

### 3. Why This Works (Theoretical Shift)
*   **Before:** Geometry was an *emergent property of the Graph*. You needed the map to navigate.
*   **After:** Geometry is an *inherent property of the Token Semantics*. The map has been memorized into the model's intuition.

By minimizing $L_{fossil}$, we force the Transformer to learn "Logical Syntax". It learns that words like "Successor" or "implies" carry specific geometric vectors that naturally position them in the hierarchy, even if no explicit link is drawn.

## Summary of Changes
1.  **Dual Forward Pass**: In every training step, we run the model *twice*: once with edges (Teacher) and once without (Student).
2.  **Self-Supervision**: The model teaches itself. The "smart" version of the model (looking at the graph) trains the "blind" version (looking at text) to be smarter.
3.  **Result**: The "OFF" mode now possesses the same navigable hyperbolic landscape as the "ON" mode, enabling zero-shot reasoning.
