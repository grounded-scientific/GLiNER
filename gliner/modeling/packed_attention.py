"""Packed-sequence attention for the GLiNER transformer wrapper.

Inference-time packing places several short requests into one row and describes which tokens may
attend to each other with a *pair attention mask* of shape ``(batch, seq_len, seq_len)``. Hugging Face
backbones do not accept such a mask directly, so this module provides:

* :func:`prepare_pair_attention_masks`, which converts a pair mask into the token / additive / boolean
  mask forms the different architectures need, and
* :func:`forward_deberta`, :func:`forward_modernbert` and :func:`forward_t5`, which drive the
  respective backbone layer by layer with the block mask applied.

All functions are stateless: they take the backbone as an argument, so they can be tested without
constructing a :class:`gliner.modeling.encoder.Transformer`.
"""

from typing import Any

import torch
from transformers.modeling_outputs import BaseModelOutput

MaskInfo = dict[str, torch.Tensor]
"""Masks produced by :func:`prepare_pair_attention_masks`."""


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of a model's parameters.

    Args:
        model: Any module.

    Returns:
        The dtype of the first parameter, or ``torch.float32`` if the module has no parameters.
    """
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _infer_device(
    pair_attention_mask: torch.Tensor,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> torch.device:
    """Pick the device masks should live on: ``input_ids``, else ``inputs_embeds``, else the pair mask."""
    if input_ids is not None:
        return input_ids.device
    if inputs_embeds is not None:
        return inputs_embeds.device
    return pair_attention_mask.device


def prepare_pair_attention_masks(
    pair_attention_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    mask_dtype: torch.dtype,
) -> MaskInfo:
    """Prepare attention masks for packed sequence processing.

    Converts pair attention masks (which specify token-to-token visibility) into the mask formats
    required by different transformer architectures. Ensures diagonal elements are attended to and
    inactive tokens are properly masked.

    Args:
        pair_attention_mask: Pairwise attention mask of shape (batch_size, seq_len, seq_len)
            where 1 indicates attention is allowed.
        attention_mask: Optional standard attention mask of shape (batch_size, seq_len).
        input_ids: Optional input token IDs for device detection.
        inputs_embeds: Optional input embeddings for device detection.
        mask_dtype: Floating dtype of the additive mask (normally the backbone's parameter dtype).

    Returns:
        Dictionary containing:
            - token_mask: Per-token mask of shape (batch_size, seq_len).
            - token_mask_bool: Boolean version of token_mask.
            - extended_mask: 4D attention mask of shape (batch_size, 1, seq_len, seq_len)
                with the dtype minimum for masked positions.
            - block_mask: Boolean 3D mask of shape (batch_size, seq_len, seq_len).
    """
    device = _infer_device(pair_attention_mask, input_ids, inputs_embeds)

    pair_mask_bool = pair_attention_mask.to(device=device, dtype=torch.bool)

    token_mask_bool = pair_mask_bool.any(dim=-1)
    if attention_mask is not None:
        token_mask_bool = token_mask_bool & attention_mask.to(device=device, dtype=torch.bool)

    seq_len = pair_mask_bool.size(-1)
    if seq_len:
        identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
        token_diag = token_mask_bool.unsqueeze(-1)
        pair_mask_bool = pair_mask_bool | (identity & token_diag)

    active = token_mask_bool.unsqueeze(-1) & token_mask_bool.unsqueeze(-2)
    pair_mask_bool = pair_mask_bool & active

    if attention_mask is not None:
        token_mask = token_mask_bool.to(attention_mask.dtype)
    else:
        token_mask = token_mask_bool.to(dtype=torch.float32)

    neg_inf = torch.finfo(mask_dtype).min
    extended_mask = (
        torch.zeros(pair_mask_bool.shape, dtype=mask_dtype, device=device)
        .masked_fill(~pair_mask_bool, neg_inf)
        .unsqueeze(1)
    )

    inactive = ~token_mask_bool
    if inactive.any():
        extended_mask = extended_mask.masked_fill(
            inactive.unsqueeze(1).unsqueeze(-1),
            torch.tensor(0.0, dtype=mask_dtype, device=device),
        )

    return {
        "token_mask": token_mask,
        "token_mask_bool": token_mask_bool,
        "extended_mask": extended_mask,
        "block_mask": pair_mask_bool,
    }


def _pop_output_flags(model_kwargs: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Remove and return ``(output_attentions, output_hidden_states, return_dict)`` from ``model_kwargs``.

    The keys are required: :meth:`gliner.modeling.encoder.Transformer.forward` always sets them.
    """
    output_attentions = model_kwargs.pop("output_attentions")
    output_hidden_states = model_kwargs.pop("output_hidden_states")
    return_dict = model_kwargs.pop("return_dict")
    return output_attentions, output_hidden_states, return_dict


def _build_output(
    last_hidden_state: torch.Tensor,
    hidden_states: tuple[torch.Tensor, ...] | None,
    attentions: tuple[torch.Tensor, ...] | None,
    return_dict: bool,
) -> BaseModelOutput | tuple:
    """Package encoder results the way Hugging Face encoders do.

    Args:
        last_hidden_state: Final-layer output of shape (batch_size, seq_len, hidden_size).
        hidden_states: Per-layer outputs, or ``None`` when not requested.
        attentions: Per-layer attention weights, or ``None`` when not requested.
        return_dict: Return a :class:`BaseModelOutput` when ``True``, otherwise a tuple that omits the
            entries that are ``None``.

    Returns:
        The packaged output.
    """
    if not return_dict:
        return tuple(v for v in (last_hidden_state, hidden_states, attentions) if v is not None)
    return BaseModelOutput(
        last_hidden_state=last_hidden_state,
        hidden_states=hidden_states,
        attentions=attentions,
    )


# --------------------------------------------------------------------------------------------
# DeBERTa
# --------------------------------------------------------------------------------------------


def _deberta_token_type_ids(input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> torch.Tensor:
    """Build all-zero ``token_type_ids`` matching whichever of ``input_ids`` / ``inputs_embeds`` is given."""
    ref = inputs_embeds if inputs_embeds is not None else input_ids
    shape = ref.size()[:-1] if inputs_embeds is not None else ref.size()
    return torch.zeros(shape, dtype=torch.long, device=ref.device)


def _deberta_append_z_steps(
    model: torch.nn.Module,
    encoded_layers: list[torch.Tensor],
    embedding_output: torch.Tensor,
    block_mask: torch.Tensor,
) -> None:
    """Append the enhanced-mask-decoder (``z_steps``) states to ``encoded_layers`` in place.

    Re-applies the last encoder layer ``model.z_steps - 1`` more times, using the second-to-last state as
    key/value input and the running state as the query, as DeBERTa's EMD does.

    Args:
        model: DeBERTa backbone with ``z_steps > 1``.
        encoded_layers: Per-layer hidden states collected so far; extended in place.
        embedding_output: Output of the embedding layer, used for relative positions.
        block_mask: Boolean (batch_size, seq_len, seq_len) visibility mask.
    """
    hidden_states = encoded_layers[-2]
    layers = [model.encoder.layer[-1] for _ in range(model.z_steps)]
    query_states = encoded_layers[-1]
    rel_embeddings = model.encoder.get_rel_embedding()
    attention_mask = model.encoder.get_attention_mask(block_mask)
    rel_pos = model.encoder.get_rel_pos(embedding_output)
    for layer in layers[1:]:
        query_states = layer(
            hidden_states,
            attention_mask,
            output_attentions=False,
            query_states=query_states,
            relative_pos=rel_pos,
            rel_embeddings=rel_embeddings,
        )
        encoded_layers.append(query_states)


def forward_deberta(
    model: torch.nn.Module,
    input_ids: torch.Tensor | None,
    model_kwargs: dict[str, Any],
    mask_info: MaskInfo,
) -> BaseModelOutput | tuple:
    """Forward pass through a DeBERTa model with packed attention support.

    Handles the specific requirements of the DeBERTa architecture including embeddings, relative
    position encodings, and optional enhanced mask decoder steps (``z_steps``).

    Args:
        model: DeBERTa-v2 backbone.
        input_ids: Input token IDs of shape (batch_size, seq_len), or None if inputs_embeds is provided.
        model_kwargs: Model keyword arguments including inputs_embeds, token_type_ids, position_ids,
            output_attentions, output_hidden_states, and return_dict. Consumed (popped) in place.
        mask_info: Masks from :func:`prepare_pair_attention_masks`.

    Returns:
        BaseModelOutput (or a tuple when ``return_dict`` is false) with last_hidden_state,
        hidden_states and attentions as requested.

    Raises:
        ValueError: If neither or both input_ids and inputs_embeds are provided.
    """
    inputs_embeds = model_kwargs.pop("inputs_embeds", None)
    token_type_ids = model_kwargs.pop("token_type_ids", None)
    position_ids = model_kwargs.pop("position_ids", None)
    output_attentions, produce_hidden, return_dict = _pop_output_flags(model_kwargs)

    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided for packed attention")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot supply both input_ids and inputs_embeds")

    if token_type_ids is None:
        token_type_ids = _deberta_token_type_ids(input_ids, inputs_embeds)

    embedding_output = model.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        mask=mask_info["token_mask"],
        inputs_embeds=inputs_embeds,
    )

    encoder_outputs = model.encoder(
        embedding_output,
        mask_info["block_mask"],
        output_hidden_states=True,
        output_attentions=output_attentions,
        return_dict=True,
    )

    encoded_layers = list(encoder_outputs.hidden_states)

    if getattr(model, "z_steps", 0) > 1:
        _deberta_append_z_steps(model, encoded_layers, embedding_output, mask_info["block_mask"])

    return _build_output(
        encoded_layers[-1],
        tuple(encoded_layers) if produce_hidden else None,
        encoder_outputs.attentions if output_attentions else None,
        return_dict,
    )


# --------------------------------------------------------------------------------------------
# ModernBERT
# --------------------------------------------------------------------------------------------


def _modernbert_geometry(
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    seq_len: int | None,
    batch_size: int | None,
) -> tuple[int, torch.device]:
    """Resolve the sequence length and device of a ModernBERT call.

    ``seq_len`` and ``batch_size`` are taken from the caller when both are given, otherwise from the
    first two dimensions of whichever of ``input_ids`` / ``inputs_embeds`` is present.
    """
    if batch_size is None or seq_len is None:
        ref = inputs_embeds if inputs_embeds is not None else input_ids
        batch_size, seq_len = ref.shape[:2]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    return seq_len, device


def _modernbert_attention_masks(
    model: torch.nn.Module,
    mask_info: MaskInfo,
    output_attentions: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ModernBERT's global and sliding-window masks and intersect both with the packing block mask.

    Args:
        model: ModernBERT backbone.
        mask_info: Masks from :func:`prepare_pair_attention_masks`.
        output_attentions: Whether attention weights will be returned.

    Returns:
        ``(global_attention_mask, sliding_window_mask)``.
    """
    token_mask_bool = mask_info["token_mask_bool"].to(torch.bool)
    model._maybe_set_compile()

    global_attention_mask, sliding_window_mask = model._update_attention_mask(
        token_mask_bool,
        output_attentions=output_attentions,
    )

    block = mask_info["block_mask"].unsqueeze(1)
    neg_inf = torch.finfo(global_attention_mask.dtype).min
    global_attention_mask = global_attention_mask.masked_fill(~block, neg_inf)
    sliding_window_mask = sliding_window_mask.masked_fill(~block, neg_inf)
    return global_attention_mask, sliding_window_mask


def _run_modernbert_layers(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, Any],
    output_attentions: bool,
    output_hidden_states: bool,
) -> tuple[torch.Tensor, tuple | None, tuple | None]:
    """Run ModernBERT's encoder layers, collecting per-layer states.

    Args:
        model: ModernBERT backbone.
        hidden_states: Embedding output.
        layer_kwargs: Keyword arguments shared by every layer (masks, positions, varlen info).
        output_attentions: Collect attention weights.
        output_hidden_states: Collect the input of every layer plus the final state.

    Returns:
        ``(final_hidden_states, all_hidden_states_or_None, all_self_attentions_or_None)``. The final norm is
        *not* applied here.
    """
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    for encoder_layer in model.layers:
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        layer_outputs = encoder_layer(hidden_states, output_attentions=output_attentions, **layer_kwargs)
        hidden_states = layer_outputs[0]
        if output_attentions and len(layer_outputs) > 1:
            all_self_attentions = (*all_self_attentions, layer_outputs[1])

    if output_hidden_states:
        all_hidden_states = (*all_hidden_states, hidden_states)
    return hidden_states, all_hidden_states, all_self_attentions


def forward_modernbert(
    model: torch.nn.Module,
    input_ids: torch.Tensor | None,
    model_kwargs: dict[str, Any],
    mask_info: MaskInfo,
) -> BaseModelOutput | tuple:
    """Forward pass through a ModernBERT model with packed attention support.

    Handles ModernBERT-specific features including global and sliding window attention patterns, and
    temporarily switches to eager attention when the model is configured for flash attention (flash
    kernels cannot take an arbitrary block mask).

    Args:
        model: ModernBERT backbone.
        input_ids: Input token IDs of shape (batch_size, seq_len), or None if inputs_embeds is provided.
        model_kwargs: Model keyword arguments including inputs_embeds, position_ids, cu_seqlens,
            max_seqlen, batch_size, seq_len, output_attentions, output_hidden_states, return_dict.
            Consumed (popped) in place.
        mask_info: Masks from :func:`prepare_pair_attention_masks`.

    Returns:
        BaseModelOutput (or a tuple when ``return_dict`` is false) with last_hidden_state,
        hidden_states and attentions as requested.

    Raises:
        ValueError: If both or neither of input_ids and inputs_embeds are provided.
    """
    inputs_embeds = model_kwargs.pop("inputs_embeds", None)
    position_ids = model_kwargs.pop("position_ids", None)
    cu_seqlens = model_kwargs.pop("cu_seqlens", None)
    max_seqlen = model_kwargs.pop("max_seqlen", None)
    batch_size = model_kwargs.pop("batch_size", None)
    seq_len = model_kwargs.pop("seq_len", None)
    output_attentions, output_hidden_states, return_dict = _pop_output_flags(model_kwargs)

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("ModernBERT requires exactly one of input_ids or inputs_embeds")

    seq_len, device = _modernbert_geometry(input_ids, inputs_embeds, seq_len, batch_size)
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    original_impl = model.config._attn_implementation
    if original_impl == "flash_attention_2":
        model.config._attn_implementation = "eager"

    global_attention_mask, sliding_window_mask = _modernbert_attention_masks(model, mask_info, output_attentions)

    hidden_states = model.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
    hidden_states, all_hidden_states, all_self_attentions = _run_modernbert_layers(
        model,
        hidden_states,
        {
            "attention_mask": global_attention_mask,
            "sliding_window_mask": sliding_window_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        },
        output_attentions,
        output_hidden_states,
    )
    hidden_states = model.final_norm(hidden_states)

    if original_impl == "flash_attention_2":
        model.config._attn_implementation = original_impl

    return _build_output(hidden_states, all_hidden_states, all_self_attentions, return_dict)


# --------------------------------------------------------------------------------------------
# T5
# --------------------------------------------------------------------------------------------


def _t5_embed_inputs(
    stack: torch.nn.Module,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Size]:
    """Return ``(inputs_embeds, input_shape)``, embedding ``input_ids`` when no embeddings were given.

    Raises:
        ValueError: If neither input_ids nor inputs_embeds is provided.
    """
    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        return stack.embed_tokens(input_ids), input_ids.size()
    return inputs_embeds, inputs_embeds.size()[:-1]


def _t5_additive_mask(block_mask: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert a boolean (batch, seq, seq) block mask into T5's additive (batch, 1, seq, seq) mask."""
    block_mask = block_mask.to(device=device, dtype=torch.bool)
    neg_inf = torch.finfo(dtype).min
    additive = torch.zeros(block_mask.shape, dtype=dtype, device=device)
    return additive.masked_fill(~block_mask, neg_inf).unsqueeze(1)


def _run_t5_blocks(
    stack: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    head_mask: Any,
    cache: dict[str, Any],
    flags: tuple[bool, bool],
) -> tuple[torch.Tensor, tuple | None, tuple | None]:
    """Run T5's encoder blocks, threading the shared relative position bias through them.

    Args:
        stack: ``T5Stack`` (the encoder).
        hidden_states: Dropout-processed input embeddings.
        attention_mask: Additive (batch, 1, seq, seq) mask.
        head_mask: Per-layer head masks, or ``None``.
        cache: ``{"use_cache", "past_key_values", "cache_position"}`` forwarded to every block.
        flags: ``(output_attentions, output_hidden_states)``.

    Returns:
        ``(hidden_states, all_hidden_states_or_None, all_attentions_or_None)``. Final layer norm and dropout
        are *not* applied here; ``all_hidden_states`` excludes the final (post-norm) state.
    """
    output_attentions, output_hidden_states = flags
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    position_bias = None

    for idx, layer_module in enumerate(stack.block):
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=head_mask[idx] if head_mask is not None else None,
            cross_attn_layer_head_mask=None,
            past_key_values=None if not cache["use_cache"] else cache["past_key_values"],
            use_cache=False,
            output_attentions=output_attentions,
            return_dict=True,
            cache_position=cache["cache_position"],
        )

        hidden_states = layer_outputs[0]
        position_bias = layer_outputs[1]

        if output_attentions:
            all_attentions = (*all_attentions, layer_outputs[2])

    return hidden_states, all_hidden_states, all_attentions


def forward_t5(
    model: torch.nn.Module,
    input_ids: torch.Tensor | None,
    model_kwargs: dict[str, Any],
    mask_info: MaskInfo,
) -> BaseModelOutput | tuple:
    """Forward pass through a T5 encoder with packed attention support.

    Handles T5/MT5-specific architecture requirements including relative position bias and proper
    attention mask formatting for the encoder stack.

    Args:
        model: ``T5EncoderModel`` (or compatible) backbone.
        input_ids: Input token IDs of shape (batch_size, seq_len), or None if inputs_embeds is provided.
            A ``input_ids`` entry in ``model_kwargs``, when present, takes precedence.
        model_kwargs: Model keyword arguments including inputs_embeds, head_mask, past_key_values,
            use_cache, output_attentions, output_hidden_states, return_dict, cache_position. Consumed
            (popped) in place.
        mask_info: Masks from :func:`prepare_pair_attention_masks`.

    Returns:
        BaseModelOutput (or a tuple when ``return_dict`` is false) with last_hidden_state,
        hidden_states and attentions as requested.

    Raises:
        ValueError: If neither input_ids nor inputs_embeds is provided, or if unsupported kwargs are
            passed.
    """
    stack = model.encoder

    kw_input_ids = model_kwargs.pop("input_ids", None)
    if input_ids is None or kw_input_ids is not None:
        input_ids = kw_input_ids

    inputs_embeds = model_kwargs.pop("inputs_embeds", None)
    head_mask = model_kwargs.pop("head_mask", None)
    past_key_values = model_kwargs.pop("past_key_values", None)
    use_cache = model_kwargs.pop("use_cache", stack.config.use_cache)
    output_attentions, output_hidden_states, return_dict = _pop_output_flags(model_kwargs)
    cache_position = model_kwargs.pop("cache_position", None)

    if model_kwargs:
        raise ValueError(f"Unsupported kwargs for T5 forward: {list(model_kwargs.keys())}")

    inputs_embeds, input_shape = _t5_embed_inputs(stack, input_ids, inputs_embeds)
    device = inputs_embeds.device

    if cache_position is None:
        cache_position = torch.arange(input_shape[1], device=device)

    attention_mask = _t5_additive_mask(mask_info["block_mask"], inputs_embeds.dtype, device)
    head_mask = stack.get_head_mask(head_mask, stack.config.num_layers)

    hidden_states, all_hidden_states, all_attentions = _run_t5_blocks(
        stack,
        stack.dropout(inputs_embeds),
        attention_mask,
        head_mask,
        {"use_cache": use_cache, "past_key_values": past_key_values, "cache_position": cache_position},
        (output_attentions, output_hidden_states),
    )

    hidden_states = stack.final_layer_norm(hidden_states)
    hidden_states = stack.dropout(hidden_states)

    if output_hidden_states:
        all_hidden_states = (*all_hidden_states, hidden_states)

    return _build_output(hidden_states, all_hidden_states, all_attentions, return_dict)
