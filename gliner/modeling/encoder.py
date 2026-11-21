from typing import Any, Dict, List, Tuple
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from .layers import LayersFuser
from ..infer_packing import InferencePackingConfig, unpack_spans, pack_requests
from .encoder_backends import (  # noqa: F401  (IS_* flags and DECODER_MODEL_MAPPING are re-exported public names)
    IS_PEFT,
    IS_LLM2VEC,
    IS_TURBOT5,
    IS_FLASHDEBERTA,
    DECODER_MODEL_MAPPING,
    build_backbone,
    apply_lora_adapter,
)
from .packed_attention import (
    MaskInfo,
    forward_t5,
    model_dtype,
    forward_deberta,
    forward_modernbert,
    prepare_pair_attention_masks,
)


class Transformer(nn.Module):
    """Flexible transformer wrapper supporting multiple architectures and configurations.

    This class provides a unified interface for various transformer models including
    encoder-only (BERT, DeBERTa), encoder-decoder (T5), and decoder-only models
    (LLaMA, Mistral) with bidirectional adaptations. It handles model initialization,
    adapter loading, and specialized forward passes for different architectures.

    Backbone selection and optional-dependency handling live in
    :mod:`gliner.modeling.encoder_backends`; the packed-attention forward passes live in
    :mod:`gliner.modeling.packed_attention`. The private ``_forward_*`` / ``_prepare_pair_attention_masks``
    methods below are thin delegates kept so existing callers and test seams keep working.

    Attributes:
        model: The underlying transformer model instance.
        layers_fuser: Optional layer fusion module when config.fuse_layers is True.
        config: Configuration object containing model hyperparameters.
    """

    def __init__(
        self,
        model_name: str,
        config: Any,
        from_pretrained: bool = False,
        labels_encoder: bool = False,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Initializes the transformer wrapper.

        Args:
            model_name: Name or path of the pretrained model to load.
            config: Configuration object containing model hyperparameters. Must have
                attributes like `encoder_config`, `labels_encoder_config`, `vocab_size`,
                `_attn_implementation`, and `fuse_layers`.
            from_pretrained: If True, loads pretrained weights. If False, initializes
                from config only. Defaults to False.
            labels_encoder: If True, initializes as a labels encoder using
                `config.labels_encoder_config`. Defaults to False.
            cache_dir: Optional directory for caching downloaded models. Defaults to None.
            local_files_only: Only load local or cached files.

        Raises:
            MissedPackageException: If required packages (llm2vec, peft) are not installed
                when needed for specific model types.
        """
        super().__init__()
        encoder_config = self._resolve_encoder_config(model_name, config, labels_encoder, cache_dir, local_files_only)

        self.model = build_backbone(model_name, encoder_config, config, from_pretrained, cache_dir, local_files_only)
        self.model = apply_lora_adapter(self.model, model_name, local_files_only)

        if config.fuse_layers:
            self.layers_fuser = LayersFuser(encoder_config.num_hidden_layers, encoder_config.hidden_size)

        if labels_encoder:
            config.labels_encoder_config = encoder_config
        else:
            config.encoder_config = encoder_config

        self.config = config

    @staticmethod
    def _resolve_encoder_config(
        model_name: str,
        config: Any,
        labels_encoder: bool,
        cache_dir: str | Path | None,
        local_files_only: bool,
    ) -> Any:
        """Return the Hugging Face config for the text or labels encoder.

        Uses ``config.labels_encoder_config`` / ``config.encoder_config`` when set, otherwise loads it
        from ``model_name`` (applying ``config.vocab_size`` when it is not ``-1``). The text encoder also
        inherits ``config._attn_implementation`` when that is set.

        Args:
            model_name: Name or path of the pretrained model.
            config: GLiNER configuration object.
            labels_encoder: Select the labels-encoder config instead of the text-encoder config.
            cache_dir: Optional download cache directory.
            local_files_only: Only use local or cached files.

        Returns:
            The resolved encoder config.
        """
        encoder_config = config.labels_encoder_config if labels_encoder else config.encoder_config
        if encoder_config is None:
            encoder_config = AutoConfig.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=local_files_only
            )
            if config.vocab_size != -1:
                encoder_config.vocab_size = config.vocab_size

        if config._attn_implementation is not None and not labels_encoder:
            encoder_config._attn_implementation = config._attn_implementation
        return encoder_config

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the transformer model.

        Handles different attention mask configurations and model architectures,
        including support for pair attention masks for packed sequences.

        Args:
            *args: Variable positional arguments passed to the model.
            **kwargs: Variable keyword arguments. Special arguments include:
                - pair_attention_mask: Optional pairwise attention mask of shape
                    (batch_size, seq_len, seq_len) for packed sequences.
                - attention_mask: Standard attention mask of shape (batch_size, seq_len).
                - input_ids: Input token IDs of shape (batch_size, seq_len).
                - Other model-specific arguments.

        Returns:
            Encoded representations of shape (batch_size, seq_len, hidden_size).
            If config.fuse_layers is True, returns fused layer outputs, otherwise
            returns the last hidden state.
        """
        pair_attention_mask = kwargs.pop("pair_attention_mask", None)
        base_attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("token_lengths", None)  # GLiNER-internal kwarg, not accepted by HF models
        input_ids, args = self._split_input_ids(args, kwargs)
        self._apply_output_defaults(kwargs)

        if pair_attention_mask is not None:
            mask_info = self._prepare_pair_attention_masks(
                pair_attention_mask,
                base_attention_mask,
                input_ids,
                kwargs.get("inputs_embeds"),
            )
            output = self._forward_packed(input_ids, args, dict(kwargs), mask_info)
        else:
            if base_attention_mask is not None:
                kwargs["attention_mask"] = base_attention_mask
            output = self.model(input_ids, *args, **kwargs)

        # Common logic for both paths
        if self.config.fuse_layers:
            return self.layers_fuser(output.hidden_states)
        return output[0]

    @staticmethod
    def _split_input_ids(args: tuple, kwargs: dict[str, Any]) -> tuple[torch.Tensor | None, tuple]:
        """Extract ``input_ids`` from ``kwargs`` (preferred) or the first positional argument.

        Args:
            args: Positional arguments given to :meth:`forward`.
            kwargs: Keyword arguments given to :meth:`forward`; ``input_ids`` is popped in place.

        Returns:
            ``(input_ids_or_None, remaining_positional_args)``.
        """
        input_ids = kwargs.pop("input_ids", None)
        if input_ids is None and args:
            return args[0], tuple(args[1:])
        return input_ids, tuple(args)

    def _apply_output_defaults(self, kwargs: dict[str, Any]) -> None:
        """Set the ``output_*`` / ``return_dict`` defaults every backbone call relies on (in place).

        ``output_hidden_states`` is forced on when layer fusion is enabled, because the fuser consumes all
        hidden states.
        """
        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("return_dict", True)
        if self.config.fuse_layers:
            kwargs["output_hidden_states"] = True
        else:
            kwargs.setdefault("output_hidden_states", False)

    def _forward_packed(
        self,
        input_ids: torch.Tensor | None,
        args: tuple,
        model_kwargs: dict[str, Any],
        mask_info: MaskInfo,
    ) -> Any:
        """Run the backbone on a packed batch, using the architecture-specific path when one exists.

        DeBERTa, ModernBERT and T5 families have dedicated implementations. Any other backbone receives
        the 4D additive mask as its ``attention_mask``.

        Args:
            input_ids: Input token IDs, or None when ``inputs_embeds`` is in ``model_kwargs``.
            args: Remaining positional arguments for the generic fallback.
            model_kwargs: Keyword arguments for the backbone (consumed by the dedicated paths).
            mask_info: Masks from :meth:`_prepare_pair_attention_masks`.

        Returns:
            The backbone output (``BaseModelOutput`` or tuple).
        """
        model_name = self.model.__class__.__name__

        if model_name in {"DebertaV2Model", "DebertaModel"}:
            return self._forward_deberta(input_ids=input_ids, model_kwargs=model_kwargs, mask_info=mask_info)
        if model_name == "ModernBertModel":
            return self._forward_modernbert(input_ids=input_ids, model_kwargs=model_kwargs, mask_info=mask_info)
        if model_name in {"T5EncoderModel", "MT5EncoderModel", "T5Model"}:
            return self._forward_t5(input_ids=input_ids, model_kwargs=model_kwargs, mask_info=mask_info)

        model_kwargs.pop("packing_config", None)
        model_kwargs["attention_mask"] = mask_info["extended_mask"]
        return self.model(*args, **model_kwargs)

    def _get_model_dtype(self) -> torch.dtype:
        """Gets the data type of the model parameters.

        Returns:
            The dtype of the model's parameters, or torch.float32 if no parameters exist.
        """
        return model_dtype(self.model)

    def _prepare_pair_attention_masks(
        self,
        pair_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        """Prepares attention masks for packed sequence processing.

        Delegates to :func:`gliner.modeling.packed_attention.prepare_pair_attention_masks`, using the
        backbone's parameter dtype for the additive mask.

        Args:
            pair_attention_mask: Pairwise attention mask of shape (batch_size, seq_len, seq_len)
                where 1 indicates attention is allowed.
            attention_mask: Optional standard attention mask of shape (batch_size, seq_len).
            input_ids: Optional input token IDs for device detection.
            inputs_embeds: Optional input embeddings for device detection.

        Returns:
            Dictionary with ``token_mask``, ``token_mask_bool``, ``extended_mask`` and ``block_mask``.
        """
        return prepare_pair_attention_masks(
            pair_attention_mask,
            attention_mask,
            input_ids,
            inputs_embeds,
            self._get_model_dtype(),
        )

    def _forward_deberta(
        self,
        input_ids: torch.Tensor | None,
        model_kwargs: Dict[str, Any],
        mask_info: Dict[str, torch.Tensor],
    ) -> BaseModelOutput:
        """Forward pass through DeBERTa models with packed attention support.

        Delegates to :func:`gliner.modeling.packed_attention.forward_deberta`.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len), or None if
                inputs_embeds is provided.
            model_kwargs: Model keyword arguments (consumed in place).
            mask_info: Masks from _prepare_pair_attention_masks.

        Returns:
            BaseModelOutput with last_hidden_state, hidden_states and attentions as requested.

        Raises:
            ValueError: If neither or both input_ids and inputs_embeds are provided.
        """
        return forward_deberta(self.model, input_ids, model_kwargs, mask_info)

    def _forward_modernbert(
        self,
        input_ids: torch.Tensor | None,
        model_kwargs: Dict[str, Any],
        mask_info: Dict[str, torch.Tensor],
    ) -> BaseModelOutput:
        """Forward pass through ModernBERT models with packed attention support.

        Delegates to :func:`gliner.modeling.packed_attention.forward_modernbert`.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len), or None if
                inputs_embeds is provided.
            model_kwargs: Model keyword arguments (consumed in place).
            mask_info: Masks from _prepare_pair_attention_masks.

        Returns:
            BaseModelOutput with last_hidden_state, hidden_states and attentions as requested.

        Raises:
            ValueError: If both or neither input_ids and inputs_embeds are provided.
        """
        return forward_modernbert(self.model, input_ids, model_kwargs, mask_info)

    def _forward_t5(
        self,
        input_ids: torch.Tensor | None,
        model_kwargs: Dict[str, Any],
        mask_info: Dict[str, torch.Tensor],
    ) -> BaseModelOutput:
        """Forward pass through T5 encoder models with packed attention support.

        Delegates to :func:`gliner.modeling.packed_attention.forward_t5`.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len), or None if
                inputs_embeds is provided.
            model_kwargs: Model keyword arguments (consumed in place).
            mask_info: Masks from _prepare_pair_attention_masks.

        Returns:
            BaseModelOutput with last_hidden_state, hidden_states and attentions as requested.

        Raises:
            ValueError: If neither input_ids nor inputs_embeds is provided, or if
                unsupported kwargs are passed.
        """
        return forward_t5(self.model, input_ids, model_kwargs, mask_info)


class Encoder(nn.Module):
    """Standard encoder module wrapping a transformer model with optional projection.

    This class provides a high-level interface for encoding text sequences, including
    support for inference-time packing to improve throughput. It handles embedding
    extraction and optional projection to a different hidden size.

    Attributes:
        bert_layer: The underlying Transformer instance.
        projection: Optional linear projection layer when config.hidden_size differs
            from the model's native hidden size.
    """

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Initializes the encoder.

        Args:
            config: Configuration object containing model hyperparameters including
                `model_name`, `hidden_size`, and transformer-specific settings.
            from_pretrained: If True, loads pretrained weights for the transformer.
                Defaults to False.
            cache_dir: Optional directory for caching downloaded models. Defaults to None.
            local_files_only: Only load local or cached files.
        """
        super().__init__()

        self.bert_layer = Transformer(
            config.model_name, config, from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only
        )

        bert_hidden_size = self.bert_layer.model.config.hidden_size

        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None) -> nn.Embedding:
        """Resizes token embeddings to accommodate new vocabulary size.

        Args:
            new_num_tokens: New vocabulary size.
            pad_to_multiple_of: Optional value to pad vocabulary size to a multiple.
                Defaults to None.

        Returns:
            The resized embedding layer.
        """
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def get_input_embeddings(self) -> nn.Embedding:
        """Gets the input embedding layer.

        Returns:
            The model's input embedding layer.
        """
        return self.bert_layer.model.get_input_embeddings()

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Encodes input text sequences into contextualized embeddings.

        Supports inference-time packing to batch multiple variable-length sequences
        efficiently when packing_config is provided and not in training mode.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len) where 1
                indicates valid tokens and 0 indicates padding.
            *args: Additional positional arguments passed to the transformer.
            **kwargs: Additional keyword arguments including:
                - packing_config: Optional InferencePackingConfig for efficient batching.
                - pair_attention_mask: Optional pairwise attention mask for packed sequences.

        Returns:
            Token embeddings of shape (batch_size, seq_len, hidden_size).
        """
        packing_config: InferencePackingConfig | None = kwargs.pop("packing_config", None)
        pair_attention_mask = kwargs.pop("pair_attention_mask", None)
        token_lengths = kwargs.pop("token_lengths", None)

        if (
            packing_config is not None
            and not self.training
            and isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and input_ids.dim() == 2
        ):
            token_embeddings = self._encode_with_packing(
                input_ids,
                attention_mask,
                packing_config,
                pair_attention_mask,
                *args,
                token_lengths=token_lengths,
                **kwargs,
            )
        else:
            bert_kwargs = dict(kwargs)
            if attention_mask is not None:
                bert_kwargs["attention_mask"] = attention_mask
            if pair_attention_mask is not None:
                bert_kwargs["pair_attention_mask"] = pair_attention_mask
            token_embeddings = self.bert_layer(
                input_ids=input_ids,
                **bert_kwargs,
            )

        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings

    def _encode_with_packing(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packing_config: InferencePackingConfig,
        pair_attention_mask: torch.Tensor | None,
        *args: Any,
        token_lengths: List[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encodes sequences using inference-time packing for efficiency.

        Packs multiple variable-length sequences into fewer, more efficient batches
        to maximize GPU utilization during inference. Short sequences are combined
        into single packed sequences.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            packing_config: Configuration for packing behavior.
            pair_attention_mask: Optional pairwise attention mask.
            *args: Additional positional arguments.
            token_lengths: Optional precomputed CPU-side per-sample token counts.
                When provided by the collator, avoids an ``attention_mask.sum().tolist()``
                GPU→CPU sync on every forward.
            **kwargs: Additional keyword arguments.

        Returns:
            Token embeddings of shape (batch_size, seq_len, hidden_size) with
            proper unpacking to restore original batch structure.
        """
        if token_lengths is not None:
            lengths = token_lengths
        else:
            lengths = attention_mask.sum(dim=-1, dtype=torch.int64).tolist()
        seq_len = int(input_ids.size(1))
        if not lengths or all(int(ln) == seq_len for ln in lengths):
            bert_kwargs = dict(kwargs)
            bert_kwargs["attention_mask"] = attention_mask
            if pair_attention_mask is not None:
                bert_kwargs["pair_attention_mask"] = pair_attention_mask
            return self.bert_layer(input_ids=input_ids, **bert_kwargs)

        # One bulk tolist() is cheaper than N per-row tolist() calls.
        all_ids = input_ids.tolist()
        requests = []
        for ids_row, length in zip(all_ids, lengths, strict=False):
            if length <= 0:
                requests.append({"input_ids": []})
            else:
                requests.append({"input_ids": ids_row[:length]})

        pad_token_id = self.bert_layer.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        packed = pack_requests(requests, packing_config, pad_token_id)

        device = input_ids.device
        packed_ids = packed.input_ids.to(device=device)
        packed_mask = packed.pair_attention_mask.to(device=device)
        packed_fallback = packed.attention_mask.to(device=device)

        attn_to_use = packed_mask if packed_mask.numel() else packed_fallback
        bert_kwargs = dict(kwargs)

        if packed_mask.numel():
            bert_kwargs["attention_mask"] = packed_fallback
            bert_kwargs["pair_attention_mask"] = packed_mask
        else:
            bert_kwargs["attention_mask"] = attn_to_use

        token_embeddings = self.bert_layer(
            input_ids=packed_ids,
            **bert_kwargs,
        )

        unpacked: List[torch.Tensor] = unpack_spans(token_embeddings, packed)
        hidden_size = token_embeddings.size(-1)
        batch, seq = input_ids.size()
        output = token_embeddings.new_zeros(batch, seq, hidden_size)
        for idx, target in enumerate(unpacked):
            tgt_len = int(target.size(0))
            if tgt_len == 0:
                continue
            output[idx, :tgt_len] = target
        return output

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            *args: Positional arguments passed to encode_text.
            **kwargs: Keyword arguments passed to encode_text.

        Returns:
            Token embeddings of shape (batch_size, seq_len, hidden_size).
        """
        token_embeddings = self.encode_text(*args, **kwargs)
        return token_embeddings


class BiEncoder(Encoder):
    """Bi-encoder architecture with separate encoders for text and labels.

    This encoder processes text sequences and label sequences through potentially
    different transformer models, producing aligned representations for both. The
    label representations are mean-pooled to create fixed-size embeddings.

    Attributes:
        bert_layer: Inherited text encoder from Encoder.
        projection: Inherited optional projection from Encoder.
        labels_encoder: Separate Transformer instance for encoding labels.
        labels_projection: Optional projection for label embeddings when label
            encoder hidden size differs from config.hidden_size.
    """

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Initializes the bi-encoder.

        Args:
            config: Configuration object containing model hyperparameters including
                `labels_encoder` (model name for label encoder) and `hidden_size`.
            from_pretrained: If True, loads pretrained weights for both encoders.
                Defaults to False.
            cache_dir: Optional directory for caching downloaded models. Defaults to None.
            local_files_only: Only load local or cached files.
        """
        super().__init__(config, from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only)
        if config.labels_encoder is not None:
            self.labels_encoder = Transformer(
                config.labels_encoder,
                config,
                from_pretrained,
                True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            le_hidden_size = self.labels_encoder.model.config.hidden_size

            if config.hidden_size != le_hidden_size:
                self.labels_projection = nn.Linear(le_hidden_size, config.hidden_size)

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Applies mean pooling over token embeddings using attention mask.

        Computes the average of token embeddings weighted by the attention mask,
        ignoring padded positions.

        Args:
            token_embeddings: Token-level embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask: Binary mask of shape (batch_size, seq_len) where 1 indicates
                valid tokens and 0 indicates padding.

        Returns:
            Pooled embeddings of shape (batch_size, hidden_size).
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_labels(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Encodes label sequences into fixed-size embeddings.

        Processes labels through the dedicated labels encoder and applies mean pooling
        to produce sentence-level representations.

        Args:
            input_ids: Label token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments (packing_config, pair_attention_mask
                and token_lengths are removed as they're not supported for labels).

        Returns:
            Pooled label embeddings of shape (batch_size, hidden_size).
        """
        label_kwargs = dict(kwargs)
        label_kwargs.pop("packing_config", None)
        label_kwargs.pop("pair_attention_mask", None)
        # token_lengths carries precomputed lengths for the *text* inputs (see
        # collator); forwarding it to the labels encoder crashes plain HF
        # backbones ("BertModel.forward() got an unexpected keyword argument
        # 'token_lengths'", #370).
        label_kwargs.pop("token_lengths", None)
        # PolyLoRA adapters select one adapter per *text* row and wrap only the text encoder.
        label_kwargs.pop("adapter_ids", None)
        label_kwargs.pop("weight_indices", None)
        label_kwargs["attention_mask"] = attention_mask
        labels_embeddings = self.labels_encoder(input_ids, *args, **label_kwargs)
        if hasattr(self, "labels_projection"):
            labels_embeddings = self.labels_projection(labels_embeddings)
        labels_embeddings = self.mean_pooling(labels_embeddings, attention_mask)
        return labels_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_input_ids: torch.Tensor | None = None,
        labels_attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the bi-encoder.

        Encodes both text sequences (token-level) and label sequences (pooled) to
        produce aligned representations.

        Args:
            input_ids: Text token IDs of shape (batch_size, seq_len).
            attention_mask: Text attention mask of shape (batch_size, seq_len).
            labels_input_ids: Label token IDs of shape (batch_size, label_seq_len).
            labels_attention_mask: Label attention mask of shape (batch_size, label_seq_len).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing:
                - token_embeddings: Text embeddings of shape (batch_size, seq_len, hidden_size).
                - labels_embeddings: Pooled label embeddings of shape (batch_size, hidden_size).
        """
        token_embeddings = self.encode_text(input_ids, attention_mask, *args, **kwargs)

        labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask, *args, **kwargs)
        return token_embeddings, labels_embeddings
