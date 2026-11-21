"""Optional-dependency detection and backbone selection for the GLiNER transformer wrapper.

This module owns everything :class:`gliner.modeling.encoder.Transformer` needs to know about
*which* backbone to build, as opposed to *how* to run it:

* detection of optional packages (``llm2vec``, ``peft``, ``turbot5``, ``flashdeberta``),
* the config-class -> backbone-class mapping, including the environment switches
  ``TURBOT5_ATTN_TYPE`` and ``USE_FLASHDEBERTA``,
* instantiation of the backbone (from pretrained weights or from a config),
* optional LoRA adapter loading through ``peft``.

The optional packages are imported through :func:`importlib.import_module` only after
:func:`~gliner.utils.is_module_available` reported them as installed, so a missing extra never breaks
``import gliner`` and static import resolvers do not mistake them for hard dependencies.
"""

import os
import warnings
import importlib
from typing import Any
from pathlib import Path

from torch import nn
from transformers import AutoModel, DebertaV2Model, T5EncoderModel

from ..utils import MissedPackageException, is_module_available

# Check for optional dependencies
IS_LLM2VEC = is_module_available("llm2vec")
IS_PEFT = is_module_available("peft")
IS_TURBOT5 = is_module_available("turbot5")
IS_FLASHDEBERTA = is_module_available("flashdeberta")


def _import_optional(module_name: str, *attributes: str) -> tuple[Any, ...]:
    """Import ``attributes`` from an optional ``module_name`` that is known to be installed.

    Args:
        module_name: Dotted module path, e.g. ``"llm2vec.models"``.
        *attributes: Names to read from that module.

    Returns:
        The requested attributes, in the order given.
    """
    module = importlib.import_module(module_name)
    return tuple(getattr(module, name) for name in attributes)


DECODER_MODEL_MAPPING: dict[str, Any]
if IS_LLM2VEC:
    _gemma, _llama, _qwen2, _mistral = _import_optional(
        "llm2vec.models", "GemmaBiModel", "LlamaBiModel", "Qwen2BiModel", "MistralBiModel"
    )
    DECODER_MODEL_MAPPING = {
        "MistralConfig": _mistral,
        "LlamaConfig": _llama,
        "GemmaConfig": _gemma,
        "Qwen2Config": _qwen2,
    }
else:
    DECODER_MODEL_MAPPING = {}

if IS_TURBOT5:
    (FlashT5EncoderModel,) = _import_optional("turbot5.model.modeling", "T5EncoderModel")

if IS_FLASHDEBERTA:
    (FlashDebertaV2Model,) = _import_optional("flashdeberta", "FlashDebertaV2Model")

if IS_PEFT:
    LoraConfig, get_peft_model = _import_optional("peft", "LoraConfig", "get_peft_model")


def _select_decoder_backbone(config_name: str) -> tuple[type, dict[str, Any], bool]:
    """Pick the bidirectional (llm2vec) backbone for a decoder-only config class.

    Args:
        config_name: Config class name present in :data:`DECODER_MODEL_MAPPING`.

    Returns:
        ``(backbone_class, from_pretrained_kwargs, is_custom)``.

    Raises:
        MissedPackageException: If ``llm2vec`` is not installed.
    """
    if not IS_LLM2VEC:
        raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
    return DECODER_MODEL_MAPPING[config_name], {}, True


def _select_t5_backbone(config: Any) -> tuple[type, dict[str, Any], bool]:
    """Pick the T5 encoder backbone, using ``turbot5`` when requested and installed.

    Side effect: when the TurboT5 backbone is chosen, ``config.encoder_config.attention_type`` is set to
    the value of ``TURBOT5_ATTN_TYPE`` so that the attention flavour is serialised with the config.

    Args:
        config: GLiNER configuration object.

    Returns:
        ``(backbone_class, from_pretrained_kwargs, is_custom)``.
    """
    turbot5_type = os.environ.get("TURBOT5_ATTN_TYPE", "basic")
    if turbot5_type and IS_TURBOT5:
        config.encoder_config.attention_type = turbot5_type
        return FlashT5EncoderModel, {"attention_type": turbot5_type}, True
    return T5EncoderModel, {}, True


def _select_deberta_backbone() -> tuple[type, dict[str, Any], bool]:
    """Pick the DeBERTa-v2 backbone, using FlashDeBERTa when ``USE_FLASHDEBERTA`` is set and installed.

    Returns:
        ``(backbone_class, from_pretrained_kwargs, is_custom)``.
    """
    if os.environ.get("USE_FLASHDEBERTA", "") and IS_FLASHDEBERTA:
        return FlashDebertaV2Model, {}, True
    return DebertaV2Model, {}, True


def select_backbone_class(config_name: str, config: Any) -> tuple[type, dict[str, Any], bool]:
    """Map a Hugging Face config class name to the backbone class GLiNER should instantiate.

    Args:
        config_name: ``encoder_config.__class__.__name__``.
        config: GLiNER configuration object (only read/mutated for the T5 branch).

    Returns:
        ``(backbone_class, from_pretrained_kwargs, is_custom)``. ``is_custom`` is ``False`` only for the
        generic :class:`~transformers.AutoModel` fallback, which must be built with ``from_config``
        instead of calling the class constructor directly.
    """
    if config_name in DECODER_MODEL_MAPPING:
        return _select_decoder_backbone(config_name)
    if config_name in {"T5Config", "MT5Config"}:
        return _select_t5_backbone(config)
    if config_name in {"DebertaV2Config"}:
        return _select_deberta_backbone()
    return AutoModel, {}, False


def build_backbone(
    model_name: str,
    encoder_config: Any,
    config: Any,
    from_pretrained: bool,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> nn.Module:
    """Instantiate the transformer backbone described by ``encoder_config``.

    Args:
        model_name: Name or path of the pretrained model.
        encoder_config: Hugging Face config of the backbone.
        config: GLiNER configuration object, forwarded to :func:`select_backbone_class`.
        from_pretrained: Load pretrained weights when ``True``; otherwise initialise from the config.
        cache_dir: Optional download cache directory.
        local_files_only: Only use local or cached files.

    Returns:
        The backbone module.
    """
    model_class, kwargs, custom = select_backbone_class(encoder_config.__class__.__name__, config)

    if from_pretrained:
        return model_class.from_pretrained(
            model_name, **kwargs, trust_remote_code=True, cache_dir=cache_dir, local_files_only=local_files_only
        )
    if not custom:
        return model_class.from_config(encoder_config, trust_remote_code=True)
    return model_class(encoder_config, **kwargs)


def apply_lora_adapter(model: nn.Module, model_name: str, local_files_only: bool) -> nn.Module:
    """Wrap ``model`` with the LoRA adapter stored next to ``model_name``, if there is one.

    A ``peft`` adapter is detected through ``<model_name>/adapter_config.json``. When an adapter is
    present but ``peft`` is not installed, a warning is emitted and ``model`` is returned unchanged.

    Args:
        model: The backbone to wrap.
        model_name: Name or path of the pretrained model.
        local_files_only: Only use local or cached files when reading the adapter config.

    Returns:
        The PEFT-wrapped model, or ``model`` itself when no adapter applies.
    """
    adapter_config_file = Path(model_name) / "adapter_config.json"
    if not adapter_config_file.exists():
        return model

    if not IS_PEFT:
        warnings.warn(
            "Adapter configs were detected, if you want to apply them you need to install peft package.",
            stacklevel=3,
        )
        return model

    adapter_config = LoraConfig.from_pretrained(model_name, local_files_only=local_files_only)
    return get_peft_model(model, adapter_config)
