# GLiNER-Relex Concept Trace (first-principles)

Target: joint zero-shot NER + relation extraction in one forward pass (`UniEncoderSpanRelexModel`).
Pinned revision: `2dc5de256b81668bbd9b3a8a8e5ce62765750551` (== local HEAD, branch `demo/three-use-cases`).

Symbols: B batch, T sub-tokens, L words, K=`max_width` (default 12, config.py:132), D hidden, C entity labels, R relation labels (`C_rel` in code), E selected entity spans, N directed pairs.

## Executive summary
1. **One sequence, three segments.** The processor builds `<<ENT>> e1 <<ENT>> e2 … <<SEP>> <<REL>> r1 <<REL>> r2 … <<SEP>> w1 … wL` (processor.py:2229-2241). One encoder pass lets every entity label, relation label and word attend to every other.
2. **Labels become vectors by position.** Hidden states at `<<ENT>>` / `<<REL>>` marker positions are the label embeddings (base.py:3206, 3274). Words are pooled separately (default first sub-token).
3. **Entities first, relations second.** Spans are scored against entity labels (base.py:3118). Spans above `threshold` become the entity set E (base.py:3040-3045). Relation pairs are built only from E.
4. **Directed pairs.** Each ordered pair (head i, tail j), i != j, is concatenated head-first and passed through one MLP (base.py:3300-3301), then dotted with each relation-label vector (base.py:3302).
5. **Two sigmoid thresholds at decode time**, plus a third inside the model that decides which spans become relation candidates (see "Thresholds").

## Deviations from the request (checked against code)
| Request said | Code says |
|---|---|
| `W_head != W_tail` as separate matrices that *guarantee* p_ab != p_ba | There is **one** shared MLP `pair_rep_layer` (base.py:2992-2994, `Linear(2D,4D)-ReLU-Dropout-Linear(4D,D)`, layers.py:77-98). Its first weight is `[W_h \| W_t]`, two independent column blocks, so asymmetry is **learned, not guaranteed**. If training drove `W_h = W_t` the pair vector would be order-invariant. |
| Prompt is `[ENT] e_k [REL] r_m [SEP] text` | Default tokens are `<<ENT>>`, `<<REL>>`, `<<SEP>>` (config.py:151-152, 453); the processor's `[ENT]`/`[SEP]` are only `getattr` fallbacks (processor.py:38-39). There are **two** `<<SEP>>`s: one after the entity block, one after the relation block (2234, 2238). Entity and relation blocks are separate, not interleaved. |
| `L = λ_E L_ent + λ_R L_rel` | Three terms when an adjacency layer is active: `span_loss_coef·L_ent + adjacency_loss_coef·L_adj + relation_loss_coef·L_rel` (base.py:3366-3370). Without adjacency: two terms (3372). |
| Single entity threshold τ_E | Two: `threshold` in `forward` selects relation-candidate spans (3151, 3045); `threshold` in `decode` filters output entities. Relation threshold is `relation_threshold` (decoder.py:1480). |
| Released checkpoint uses all-pairs | **Not verified.** `knowledgator/gliner-relex-large-v1.0`'s `config.json` was not fetched (no web access in this run). In code, all-pairs is `relations_layer="none"` (config.py:468-469; base.py:3293). |
| Entity vectors `s_a` = encoder states | They are `span_rep_layer(words, span_idx)` outputs (base.py:3117), i.e. a learned span head over pooled word states, not raw hidden states. Default `span_mode="markerV0"` (config.py:137). |

## Execution table
| Step | Plain-English role | Function & `file:line` | Math / operation | Input type / shape | Output type / shape | Why it matters |
|---|---|---|---|---|---|---|
| 1. Build prompt | Prepend entity labels, then relation labels, to each text | `RelationExtractionSpanProcessor.prepare_inputs` processor.py:2187-2243 | `X = [<<ENT>> e_1 … <<SEP>> <<REL>> r_1 … <<SEP>> w_1 … w_L]` | `texts: List[List[str]]`, entity/relation label lists | `input_texts: List[List[str]]`, `prompt_lengths: List[int]` | Label count enlarges the sequence, so latency grows with C+R. |
| 2. Tokenize + word mask | Sub-tokenize, mark which sub-token starts each text word | `tokenize_inputs` processor.py:2245-2270 (mask at 2267) | `words_mask[t] = i` for the first sub-token of word i, 0 over the prompt | `is_split_into_words=True`, `padding="longest"` | `input_ids [B,T]`, `attention_mask [B,T]`, `words_mask [B,T]` | Maps predictions back to word indices, hence to character offsets. |
| 3. Joint encode | Full bidirectional attention over prompt and text | `token_rep_layer` call base.py:3183 | `H = Encoder(X)` | `input_ids [B,T]`, `attention_mask [B,T]` | `token_embeds [B,T,D]` | Labels are contextualized by the text and vice versa. This is why the model is a uni-encoder. |
| 4. Pool prompts and words | Read label vectors at marker positions; pool words | `_extract_prompt_features_and_word_embeddings` base.py:479, called 3206; `extract_prompt_features` utils.py:175 | `P_ent = H[pos(<<ENT>>)]`; `W = pool_first(H, words_mask)` | `token_embeds [B,T,D]` | `prompts_embedding [B,C,D]` + mask, `words_embedding [B,L,D]` + mask | Marker-position pooling gives label vectors with no separate label encoder. |
| 5. Relation label vectors | Same pooling at `<<REL>>` positions | `extract_prompt_features(config.rel_token_index, …)` base.py:3274-3282 | `P_rel = H[pos(<<REL>>)]` | `token_embeds [B,T,D]` | `rel_prompts_embedding [B,R,D]` + mask | Note: entity prompts go through `prompt_rep_layer` (3234); relation prompts do **not** (no projection before 3302). |
| 6. Entity prompt projection | Project entity label vectors | `prompt_rep_layer` base.py:3234 | `P_ent' = MLP(P_ent)` | `[B,C,D]` | `[B,C,D]` | Entity head is trained on a projected space. |
| 7. Span vectors + NER scores | Build a vector for every (start, width) span and score against entity labels | `represent_spans` base.py:3104-3134 (einsum 3118) | `S = SpanRep(W, span_idx)`; `O[b,l,k,c] = <S[b,l,k], P_ent'[b,c]>` | `words_embedding [B,L,D]`, `span_idx [B,L*K,2]` | `scores [B,L,K,C]`, `S [B,L,K,D]` | Same span head as plain GLiNER. |
| 8. Select entities | Keep spans whose best-label probability exceeds the threshold; pack them | `select_span_target_embedding` base.py:2996-3066 (keep 3040-3045); `select_target_embedding` 3068-3102 | `keep = max_c σ(O) > τ ∧ span_mask`; training uses gold: `max_c y == 1` (3037-3039) | `S [B,L*K,D]`, `scores [B,L,K,C]` | `target_span_rep [B,E,D]`, `target_span_mask [B,E]`, `entity_spans [B,E,2]` | Relation cost depends on E, not on L*K. A missed entity here can never be recovered by relation scoring. |
| 9a. Pair selection: all-pairs | Enumerate every ordered pair of valid entities | `build_all_entity_pairs` utils.py:411-470; called base.py:3293 | `{(i,j): i≠j, i,j<n_b}`, `N = max_b n_b(n_b-1)` | `[B,E,D]`, `span_mask [B,E]` | `pair_idx [B,N,2]` (-1 pad), `pair_mask [B,N]`, `head_rep, tail_rep [B,N,D]` | O(E²) pairs, built with a Python loop per batch row. |
| 9b. Pair selection: adjacency-guided | Predict an E×E edge matrix, keep pairs above threshold | `RelationsRepLayer.forward` relations_layers.py:388-404 (call base.py:3267); `build_entity_pairs` utils.py:326-408 (call base.py:3289) | `Â = σ(f(X))`, masked by `Â·m_i·m_j` (relations_layers.py:42); keep `Â[i,j] > τ_A` for i≠j | `X [B,E,D]`, `mask [B,E]` | `Â [B,E,E] ∈ [0,1]`, then same four tensors as 9a | Prunes pairs before the expensive scoring. Training selects with the **gold** adjacency (base.py:3288). |
| 10. Pair projection | Concatenate head then tail, project | base.py:3300-3301 (`pair_rep_layer` built 2992) | `p_{a,b} = W₂ ReLU(W₁[s_a ; s_b] + b₁) + b₂`, `W₁ = [W_h \| W_t] ∈ R^{4D×2D}`, dropout between | `head_rep, tail_rep [B,N,D]` | `pair_rep [B,N,D]` | Order is encoded by slot position in the concatenation. |
| 11a. Score: dot | Pair vector against each relation label | base.py:3302 | `r[b,n,m] = <p_n, P_rel[b,m]>` | `pair_rep [B,N,D]`, `rel_prompts_embedding [B,R,D]` | `pair_scores [B,N,R]` | One einsum, no per-label expansion. |
| 11b. Score: triples | Knowledge-graph interaction over (h, r, t) | base.py:3306-3316; `TriplesScoreLayer` triples_layers.py:557 | e.g. TransE: `-‖h + r − t‖`; modes UM, SE, TransE, … | `h,t [B,N,R,D]`, `r [B,N,R,D]` flattened to `[B·N·R, D]` | `pair_scores [B,N,R]` | Materializes `B·N·R·D` floats: memory-heavy. Replaces `pair_rep_layer`. |
| 12. Joint loss | Entity + (adjacency) + relation terms | base.py:3321-3372; `rel_loss` 3464-3537; `adj_loss` 3399-3462 | `L = c_s·L_ent + c_a·L_adj + c_r·L_rel` | logits, labels, masks | scalar `loss` | Separate focal params for relations: `rel_alpha`, `rel_gamma` (3355-3356). |
| 13. Output | Bundle logits and pair tensors | `GLiNERRelexOutput` base.py:3379-3397 | inference only: `rel_idx`, `rel_logits`, `rel_mask`, `entity_spans` | – | `GLiNERRelexOutput` | Pair tensors are dropped when `labels` are given (3386-3395). |
| 14. Decode entities | Threshold and resolve spans | `SpanRelexDecoder.decode` decoder.py:1470-1538 | `σ(calibrated logits) > τ_E` | `logits [B,L,K,C]` | `spans: List[List[(start,end,label,score)]]` | Greedy resolution may drop or reorder entities. |
| 15. Decode relations | Map model pair indices to decoded spans and threshold | `_decode_relations` decoder.py:1342-1468; vectorized variant `_decode_relations_batch` 1145-1232 | `p = σ((r − b)/T)`; keep `p > τ_R` (calibration: calibration.py:59) | `rel_idx [B,N,2]`, `rel_logits [B,N,R]`, `rel_mask [B,N]`, `entity_spans` | `List[List[DecodedRelation(head, label, tail, score)]]` | `_build_entity_span_to_decoded_idx` (1404) re-maps indices; pairs whose entity was dropped are skipped (1433-1434). |

## Asymmetry, from first principles
Let `s_a, s_b ∈ R^D` and `W₁ = [W_h | W_t]`. Then `W₁[s_a; s_b] = W_h s_a + W_t s_b`. Swapping gives `W_h s_b + W_t s_a`. These agree for all pairs only if `W_h = W_t` (or the difference lies in a null space). Nothing in the code constrains it, so direction is **learned from the supervision** (relation labels in `rel_matrix` are directed: head index then tail index, `pair_idx[...,0]` head and `[...,1]` tail, utils.py:380, 447-448).

Contrast: the `bilinear` adjacency decoder computes `σ(Z Zᵀ)` (relations_layers.py:180), which is **symmetric**: `Â[i,j] = Â[j,i]`. `dot` (66-67) is also symmetric, and `gcn` / `gat` end in `σ(Z Zᵀ)` (266, 309), so they are symmetric too. The directed adjacency modes are `mlp` (105: `cat([Xi, Xj])`) and `attention` (141: row-normalized attention weights, generally asymmetric). With a symmetric adjacency, the matrix only prunes pairs; direction is then decided by `pair_rep_layer`.

## Thresholds (the "τ_E, τ_R" the request asks about)
| Name | Where | Default | Effect |
|---|---|---|---|
| `threshold` (forward) | base.py:3151 → 3045 | 0.5 | Which spans become relation-candidate entities |
| `adjacency_threshold` | base.py:3152 → 3290 | 0.5 | Which pairs survive (adjacency mode only) |
| `threshold` (decode) | decoder.py:1479 | 0.5 | Which spans are returned as entities |
| `relation_threshold` | decoder.py:1480 | 0.5 | Which (pair, label) scores are returned |

Two entity thresholds can disagree: a span can be a relation endpoint (passes forward) yet be absent from the entity output (fails decode), in which case its relations are skipped at decoder.py:1433-1434.

## Findings worth a second look (not bugs proven by this trace)
- **Gold-vs-predicted exposure gap.** Training selects entities with gold labels (base.py:3037-3039) and, in adjacency mode, gold adjacency (3288); inference uses predictions. Relation scoring is therefore never trained on predicted-entity noise.
- **Index alignment is patched, not guaranteed.** `rel_matrix` is truncated or zero-padded to N and R (base.py:3331-3349); the comment says mismatches happen "in edge cases".
- **Unprojected relation labels.** Entity labels pass `prompt_rep_layer`; relation labels do not (compare 3234 and 3302).
- **Training-time loss without relations.** If `rel_matrix` is absent, `loss` stays the raw entity loss, not multiplied by `span_loss_coef` (3322-3325 vs 3360).

## Cost (derived from reading, not measured)
- Encoder: O((C + R + L)²) attention (labels in sequence).
- Span scoring: O(B·L·K·C·D).
- Pairs: N = E(E−1) all-pairs, or the count above τ_A with adjacency; pair MLP about 12·D² multiply-adds per pair (`2D→4D` is 8D², `4D→D` is 4D²); relation dot O(B·N·R·D). Triples path: O(B·N·R·D) memory.

## Verification metadata
- RubberDuck MCP session: `558f11d2f005414c9c4954cf8dcc7887`. Local-path loading is refused by the server, so the repo was loaded **remotely** as `urchade/GLiNER` at the exact SHA above (`commit_pinned`, `loaded_commit` matches local HEAD). Local untracked files (`rubberduck-concept-trace/`, `examples/demo_three_use_cases/`) are not part of that graph.
- `analysis_id`s: `urchade__GLiNER__gliner_modeling_base_py`, `…_modeling_utils_py`, `…_modeling_multitask_relations_layers_py`, `gliner_config_py`, `gliner_data_processing_processor_py`, `gliner_decoding_decoder_py`.
- **`coherence: 1.0` confirmed** (evidence_source `graph`) for `call_chain(build_entity_pairs)`: sole non-test caller `UniEncoderSpanRelexModel.forward` at base.py:3289; resolver coverage complete. Also 1.0 for `trace_variable(pair_rep)`: 3300 → 3301 → 3302 → 3304 (→ `relation_embeddings` → output 3379/3393).
- **Partial:** `call_chain(forward)` returned coherence 1.0 but `query_outcome: partial` (79 name-matched references unresolved; callees are merged across the class hierarchy, so the callee list mixes several models' `forward`). The Relex-specific chain above is taken from `read_source` of base.py:3136-3397, not from that list.
- **Not verified:** the `knowledgator/gliner-relex-large-v1.0` checkpoint config; token strings and `relations_layer` actually stored there; the "130x" or other speed claims; all complexity figures are derived.
- `triples_layers.py` interaction formulas were read only at the `TriplesScoreLayer` wrapper level (557-597); individual interaction math (UM/SE/TransE) is quoted from class names, not re-derived.
