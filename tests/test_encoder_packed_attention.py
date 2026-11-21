"""Characterization tests for ``gliner.modeling.encoder.Transformer``.

These lock the behaviour of the packed-attention forward paths (DeBERTa, ModernBERT, T5), the
attention-mask preparation, backbone-class selection and the public import surface of the encoder
module, so that the module can be restructured without changing behaviour.
"""

from types import SimpleNamespace

import torch
import pytest
from transformers import T5Config, BertConfig, DebertaV2Config, ModernBertModel, ModernBertConfig
from transformers.models.t5.modeling_t5 import T5Stack

from gliner.modeling import encoder
from gliner.modeling.encoder import Transformer

# The packed T5 / ModernBERT paths call private transformers APIs. Newer transformers releases removed
# them, which makes those paths raise AttributeError. That is existing behaviour and out of scope for the
# structural refactor these tests protect, so the affected cases are strict-xfail: they run in full on
# transformers versions that still ship the APIs and will flag (XPASS) once the paths are repaired.
T5_PACKED_BROKEN = not hasattr(T5Stack, "get_head_mask")
MODERNBERT_PACKED_BROKEN = not (
    hasattr(ModernBertModel, "_maybe_set_compile") and hasattr(ModernBertModel, "_update_attention_mask")
)
xfail_t5 = pytest.mark.xfail(
    T5_PACKED_BROKEN, raises=AttributeError, strict=True, reason="T5Stack.get_head_mask removed in this transformers"
)
xfail_modernbert = pytest.mark.xfail(
    MODERNBERT_PACKED_BROKEN,
    raises=AttributeError,
    strict=True,
    reason="ModernBertModel._maybe_set_compile/_update_attention_mask removed in this transformers",
)
ARCH_PARAMS = [
    "deberta",
    pytest.param("t5", marks=xfail_t5),
    pytest.param("modernbert", marks=xfail_modernbert),
]

HIDDEN = 32
SEQ = 6


def _deberta_config():
    return DebertaV2Config(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=8,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def _t5_config():
    return T5Config(
        vocab_size=64,
        d_model=HIDDEN,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
    )


def _modernbert_config():
    return ModernBertConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        global_attn_every_n_layers=2,
        local_attention=4,
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )


CONFIG_FACTORIES = {
    "deberta": _deberta_config,
    "t5": _t5_config,
    "modernbert": _modernbert_config,
}


def _build(arch: str, fuse_layers: bool = False) -> Transformer:
    torch.manual_seed(0)
    encoder_config = CONFIG_FACTORIES[arch]()
    config = SimpleNamespace(
        encoder_config=encoder_config,
        labels_encoder_config=None,
        vocab_size=-1,
        _attn_implementation=None,
        fuse_layers=fuse_layers,
    )
    transformer = Transformer("not-a-real-model", config, from_pretrained=False)
    transformer.eval()
    return transformer


def _packed_inputs(lengths=(3, 3)):
    """Two sequences packed in one row, with a block-diagonal pair mask."""
    total = sum(lengths)
    torch.manual_seed(1)
    input_ids = torch.randint(5, 60, (1, total))
    pair = torch.zeros(1, total, total, dtype=torch.long)
    start = 0
    for length in lengths:
        pair[0, start : start + length, start : start + length] = 1
        start += length
    return input_ids, torch.ones(1, total, dtype=torch.long), pair


# --------------------------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------------------------


def test_public_names_importable():
    for name in (
        "Transformer",
        "Encoder",
        "BiEncoder",
        "IS_LLM2VEC",
        "IS_PEFT",
        "IS_TURBOT5",
        "IS_FLASHDEBERTA",
        "DECODER_MODEL_MAPPING",
    ):
        assert hasattr(encoder, name), name
    assert isinstance(encoder.DECODER_MODEL_MAPPING, dict)


# --------------------------------------------------------------------------------------------
# Backbone selection / construction
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "expected_class"),
    [("deberta", "DebertaV2Model"), ("t5", "T5EncoderModel"), ("modernbert", "ModernBertModel")],
)
def test_backbone_class_selection(arch, expected_class):
    transformer = _build(arch)
    assert transformer.model.__class__.__name__ == expected_class


def test_state_dict_keys_are_stable_for_checkpoints():
    transformer = _build("deberta", fuse_layers=True)
    keys = list(transformer.state_dict())
    assert any(k.startswith("model.embeddings.") for k in keys)
    assert any(k.startswith("layers_fuser.") for k in keys)
    assert all(k.startswith(("model.", "layers_fuser.")) for k in keys)


def test_config_is_written_back_for_text_and_labels_encoder():
    encoder_config = _deberta_config()
    config = SimpleNamespace(
        encoder_config=None,
        labels_encoder_config=encoder_config,
        vocab_size=-1,
        _attn_implementation="eager",
        fuse_layers=False,
    )
    Transformer("x", config, from_pretrained=False, labels_encoder=True)
    assert config.labels_encoder_config is encoder_config
    assert config.encoder_config is None


# --------------------------------------------------------------------------------------------
# Attention-mask preparation
# --------------------------------------------------------------------------------------------


def test_prepare_pair_attention_masks_block_diagonal():
    transformer = _build("deberta")
    input_ids, attention_mask, pair = _packed_inputs((2, 2))
    info = transformer._prepare_pair_attention_masks(pair, attention_mask, input_ids, None)

    expected_block = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]], dtype=torch.bool)
    assert torch.equal(info["block_mask"], expected_block)
    assert torch.equal(info["token_mask_bool"], torch.ones(1, 4, dtype=torch.bool))
    assert info["token_mask"].dtype == attention_mask.dtype
    assert info["extended_mask"].shape == (1, 1, 4, 4)

    neg_inf = torch.finfo(next(transformer.model.parameters()).dtype).min
    assert torch.all(info["extended_mask"][0, 0][expected_block[0]] == 0)
    assert torch.all(info["extended_mask"][0, 0][~expected_block[0]] == neg_inf)


def test_prepare_pair_attention_masks_padding_and_diagonal():
    transformer = _build("deberta")
    # Row 3 is padding: pair mask row is all zero and attention mask is 0.
    pair = torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
    attention_mask = torch.tensor([[1, 1, 0]])
    info = transformer._prepare_pair_attention_masks(pair, attention_mask, None, None)

    # Token 1 has no visible token in ``pair`` but is active in ``attention_mask`` -> token_mask only
    # comes from the pair mask, so it is inactive and the diagonal must not be forced for it.
    assert info["token_mask_bool"].tolist() == [[True, False, False]]
    assert info["block_mask"].tolist() == [[[True, False, False], [False, False, False], [False, False, False]]]
    # Inactive rows are fully unmasked (zeros) to avoid NaNs in softmax.
    assert torch.all(info["extended_mask"][0, 0, 1:, :] == 0)
    assert info["token_mask"].dtype == attention_mask.dtype


def test_prepare_pair_attention_masks_without_attention_mask_uses_float_token_mask():
    transformer = _build("deberta")
    pair = torch.ones(1, 3, 3, dtype=torch.long)
    info = transformer._prepare_pair_attention_masks(pair, None, None, None)
    assert info["token_mask"].dtype == torch.float32
    assert info["block_mask"].all()


def test_prepare_pair_attention_masks_device_comes_from_inputs_embeds():
    transformer = _build("deberta")
    pair = torch.ones(1, 3, 3, dtype=torch.long)
    info = transformer._prepare_pair_attention_masks(pair, None, None, torch.zeros(1, 3, HIDDEN))
    assert info["extended_mask"].device == pair.device


# --------------------------------------------------------------------------------------------
# Forward: non-packed path
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["deberta", "t5", "modernbert"])
def test_forward_without_pair_mask_matches_backbone(arch):
    transformer = _build(arch)
    torch.manual_seed(2)
    input_ids = torch.randint(5, 60, (2, SEQ))
    attention_mask = torch.ones(2, SEQ, dtype=torch.long)
    attention_mask[1, -2:] = 0

    with torch.no_grad():
        out = transformer(input_ids=input_ids, attention_mask=attention_mask)
        ref = transformer.model(input_ids=input_ids, attention_mask=attention_mask)[0]
    assert torch.allclose(out, ref, atol=1e-6)


def test_forward_accepts_positional_input_ids_and_drops_token_lengths():
    transformer = _build("deberta")
    input_ids = torch.randint(5, 60, (1, SEQ))
    with torch.no_grad():
        a = transformer(input_ids, token_lengths=[SEQ])
        b = transformer(input_ids=input_ids)
    assert torch.allclose(a, b, atol=1e-6)


def test_forward_with_fuse_layers_uses_all_hidden_states():
    transformer = _build("deberta", fuse_layers=True)
    input_ids = torch.randint(5, 60, (1, SEQ))
    with torch.no_grad():
        out = transformer(input_ids=input_ids)
    assert out.shape == (1, SEQ, HIDDEN)


# --------------------------------------------------------------------------------------------
# Forward: packed path (segments packed in one row must not see each other)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCH_PARAMS)
def test_packed_segments_are_isolated(arch):
    """Changing the tokens of one packed segment must not change the other segment's output."""
    transformer = _build(arch)
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    perturbed = input_ids.clone()
    perturbed[:, 3:] = (perturbed[:, 3:] + 7) % 60 + 3

    with torch.no_grad():
        base = transformer(input_ids=input_ids, attention_mask=attention_mask, pair_attention_mask=pair)
        other = transformer(input_ids=perturbed, attention_mask=attention_mask, pair_attention_mask=pair)

    assert base.shape == (1, 6, HIDDEN)
    assert torch.isfinite(base).all()
    assert torch.allclose(base[:, :3], other[:, :3], atol=1e-5)
    assert not torch.allclose(base[:, 3:], other[:, 3:], atol=1e-5)


@pytest.mark.parametrize("arch", ARCH_PARAMS)
def test_packed_forward_with_fuse_layers_returns_fused_states(arch):
    transformer = _build(arch, fuse_layers=True)
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    with torch.no_grad():
        out = transformer(input_ids=input_ids, attention_mask=attention_mask, pair_attention_mask=pair)
    assert out.shape == (1, 6, HIDDEN)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCH_PARAMS)
def test_packed_forward_is_deterministic(arch):
    """Two identical calls give identical output (no hidden state is mutated between calls)."""
    transformer = _build(arch)
    input_ids, attention_mask, pair = _packed_inputs((2, 4))
    with torch.no_grad():
        a = transformer(input_ids=input_ids, attention_mask=attention_mask, pair_attention_mask=pair)
        b = transformer(input_ids=input_ids, attention_mask=attention_mask, pair_attention_mask=pair)
    assert torch.equal(a, b)


@xfail_modernbert
def test_modernbert_restores_attention_implementation():
    transformer = _build("modernbert")
    transformer.model.config._attn_implementation = "flash_attention_2"
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    try:
        with torch.no_grad():
            transformer(input_ids=input_ids, attention_mask=attention_mask, pair_attention_mask=pair)
    finally:
        restored = transformer.model.config._attn_implementation
    assert restored == "flash_attention_2"


# --------------------------------------------------------------------------------------------
# Per-architecture helpers: output container + error paths
# --------------------------------------------------------------------------------------------


def _mask_info(transformer, input_ids, attention_mask, pair):
    return transformer._prepare_pair_attention_masks(pair, attention_mask, input_ids, None)


def _model_kwargs(**overrides):
    kwargs = {"output_attentions": False, "output_hidden_states": True, "return_dict": True}
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("arch", "method"),
    [
        ("deberta", "_forward_deberta"),
        pytest.param("t5", "_forward_t5", marks=xfail_t5),
        pytest.param("modernbert", "_forward_modernbert", marks=xfail_modernbert),
    ],
)
def test_forward_helpers_return_dict_and_tuple_layouts(arch, method):
    transformer = _build(arch)
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    info = _mask_info(transformer, input_ids, attention_mask, pair)
    run = getattr(transformer, method)

    with torch.no_grad():
        as_dict = run(input_ids=input_ids, model_kwargs=_model_kwargs(), mask_info=info)
        as_tuple = run(input_ids=input_ids, model_kwargs=_model_kwargs(return_dict=False), mask_info=info)
        no_hidden = run(input_ids=input_ids, model_kwargs=_model_kwargs(output_hidden_states=False), mask_info=info)

    assert as_dict.last_hidden_state.shape == (1, 6, HIDDEN)
    assert as_dict.hidden_states is not None
    assert len(as_dict.hidden_states) == transformer.model.config.num_hidden_layers + 1 or arch == "t5"
    assert as_dict.attentions is None

    assert isinstance(as_tuple, tuple)
    assert torch.equal(as_tuple[0], as_dict.last_hidden_state)
    assert len(as_tuple) == 2  # last_hidden_state + hidden_states
    assert len(as_tuple[1]) == len(as_dict.hidden_states)

    assert no_hidden.hidden_states is None


def test_deberta_helper_validates_inputs():
    transformer = _build("deberta")
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    info = _mask_info(transformer, input_ids, attention_mask, pair)

    with pytest.raises(ValueError, match="Either input_ids or inputs_embeds"):
        transformer._forward_deberta(input_ids=None, model_kwargs=_model_kwargs(), mask_info=info)
    with pytest.raises(ValueError, match="Cannot supply both"):
        transformer._forward_deberta(
            input_ids=input_ids,
            model_kwargs=_model_kwargs(inputs_embeds=torch.zeros(1, 6, HIDDEN)),
            mask_info=info,
        )


def test_deberta_helper_accepts_inputs_embeds_and_z_steps():
    transformer = _build("deberta")
    transformer.model.z_steps = 2
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    info = _mask_info(transformer, input_ids, attention_mask, pair)
    with torch.no_grad():
        embeds = transformer.model.embeddings.word_embeddings(input_ids)
        out = transformer._forward_deberta(
            input_ids=None, model_kwargs=_model_kwargs(inputs_embeds=embeds), mask_info=info
        )
    # z_steps=2 appends one extra enhanced-mask-decoder state.
    assert len(out.hidden_states) == transformer.model.config.num_hidden_layers + 2


def test_modernbert_helper_validates_inputs():
    transformer = _build("modernbert")
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    info = _mask_info(transformer, input_ids, attention_mask, pair)
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        transformer._forward_modernbert(input_ids=None, model_kwargs=_model_kwargs(), mask_info=info)
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        transformer._forward_modernbert(
            input_ids=input_ids,
            model_kwargs=_model_kwargs(inputs_embeds=torch.zeros(1, 6, HIDDEN)),
            mask_info=info,
        )


def test_t5_helper_validates_inputs_and_rejects_unknown_kwargs():
    transformer = _build("t5")
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    info = _mask_info(transformer, input_ids, attention_mask, pair)
    with pytest.raises(ValueError, match="Either input_ids or inputs_embeds"):
        transformer._forward_t5(input_ids=None, model_kwargs=_model_kwargs(), mask_info=info)
    with pytest.raises(ValueError, match="Unsupported kwargs for T5 forward"):
        transformer._forward_t5(input_ids=input_ids, model_kwargs=_model_kwargs(bogus=1), mask_info=info)


@xfail_t5
def test_t5_helper_keyword_input_ids_overrides_positional():
    transformer = _build("t5")
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    info = _mask_info(transformer, input_ids, attention_mask, pair)
    other = torch.randint(5, 60, input_ids.shape)
    with torch.no_grad():
        via_kwarg = transformer._forward_t5(
            input_ids=input_ids, model_kwargs=_model_kwargs(input_ids=other), mask_info=info
        )
        direct = transformer._forward_t5(input_ids=other, model_kwargs=_model_kwargs(), mask_info=info)
    assert torch.equal(via_kwarg.last_hidden_state, direct.last_hidden_state)


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="existing behaviour: the generic packed fallback pops input_ids and never forwards it to the backbone",
)
def test_generic_backbone_falls_back_to_extended_mask():
    """Backbones without a dedicated packed path receive the 4D additive mask as ``attention_mask``."""
    config = SimpleNamespace(
        encoder_config=BertConfig(
            vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
        ),
        labels_encoder_config=None,
        vocab_size=-1,
        _attn_implementation="eager",
        fuse_layers=False,
    )
    transformer = Transformer("x", config, from_pretrained=False).eval()
    input_ids, attention_mask, pair = _packed_inputs((3, 3))
    with torch.no_grad():
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_attention_mask=pair,
            packing_config=object(),  # must be dropped before reaching the backbone
        )
    assert out.shape == (1, 6, HIDDEN)
