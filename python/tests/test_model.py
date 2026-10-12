"""独立期望值与行为测试：错误 scale/mask/LN/teacher forcing 必须被发现。"""

import math

import pytest
import torch
from transformer_lab import model as model_api


def small_model(**kwargs):
    torch.manual_seed(7)
    return model_api.Transformer(model_api.TransformerConfig(vocab_size=8, **kwargs))


def test_attention_matches_hand_calculated_formula():
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.0, 3.0], [10.0, 7.0]]]], dtype=torch.float64)
    allowed = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    output, weights = model_api.scaled_dot_product_attention(q, k, v, allowed)
    p = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    expected = torch.tensor([[[[2 * p + 10 * (1 - p), 3 * p + 7 * (1 - p)]]]])
    torch.testing.assert_close(output, expected.double(), atol=5e-7, rtol=0)
    torch.testing.assert_close(weights, torch.tensor([[[[p, 1 - p]]]]).double())


def test_attention_mask_blocks_values_and_gradients():
    q = torch.ones(1, 1, 1, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[[[3.0, 4.0], [1e8, -1e8]]]], requires_grad=True)
    output, weights = model_api.scaled_dot_product_attention(
        q, k, v.double(), torch.tensor([[[[True, False]]]])
    )
    torch.testing.assert_close(output, torch.tensor([[[[3.0, 4.0]]]]).double())
    assert weights[0, 0, 0, 1] == 0
    output.sum().backward()
    assert torch.count_nonzero(v.grad[0, 0, 1]) == 0
    assert torch.count_nonzero(k.grad[0, 0, 1]) == 0


def test_attention_rejects_fully_masked_rows():
    x = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="mask|屏蔽"):
        model_api.scaled_dot_product_attention(x, x, x, torch.zeros(2, 2, dtype=torch.bool))


def test_attention_and_layernorm_gradcheck():
    torch.manual_seed(4)
    q = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[[True, True, False], [True, True, True]]]])
    assert torch.autograd.gradcheck(
        lambda q, k, v: model_api.scaled_dot_product_attention(q, k, v, mask)[0], (q, k, v)
    )
    norm = model_api.ManualLayerNorm(2, 1e-5)
    assert torch.autograd.gradcheck(norm, (q,))


def test_layernorm_uses_population_variance_and_eps_inside_root():
    norm = model_api.ManualLayerNorm(2, eps=0.5)
    x = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(
        norm(x), torch.tensor([[-1 / math.sqrt(1.5), 1 / math.sqrt(1.5)]]).double()
    )


def test_sinusoidal_encoding_including_odd_width():
    pe = model_api.sinusoidal_position_encoding(3, 5)
    assert pe.shape == (3, 5)
    assert pe.dtype == torch.float64
    torch.testing.assert_close(pe[0], torch.tensor([0, 1, 0, 1, 0]).double())
    assert pe[1, 0].item() == pytest.approx(math.sin(1), abs=1e-14)
    assert pe[1, 1].item() == pytest.approx(math.cos(1), abs=1e-14)
    assert pe[1, 4].item() == pytest.approx(math.sin(10000 ** (-4 / 5)), abs=1e-14)


def test_decoder_cannot_observe_future_tokens():
    model = small_model(num_encoder_layers=2, num_decoder_layers=2)
    src = torch.tensor([[3, 4, 2, 0]])
    a = model(src, torch.tensor([[1, 3, 4, 5]]))
    b = model(src, torch.tensor([[1, 3, 6, 7]]))
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-13, rtol=1e-13)
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_source_padding_keys_do_not_affect_cross_attention():
    model = small_model()
    src = torch.tensor([[3, 4, 0, 0]])
    tgt = torch.tensor([[1, 3, 4]])
    before, trace = model.forward_with_trace(src, tgt)
    with torch.no_grad():
        model.src_embedding.weight[0].fill_(1e6)
    after, changed_trace = model.forward_with_trace(src, tgt)
    torch.testing.assert_close(before, after, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        trace["encoder_output"][:, :2], changed_trace["encoder_output"][:, :2]
    )
    assert not torch.allclose(
        trace["encoder_output"][:, 2:], changed_trace["encoder_output"][:, 2:]
    )


def test_target_padding_key_does_not_affect_later_nonpad_query():
    model = small_model()
    src, tgt = torch.tensor([[3, 4, 2]]), torch.tensor([[1, 0, 4]])
    before = model(src, tgt)
    with torch.no_grad():
        model.tgt_embedding.weight[0].fill_(1e6)
    after = model(src, tgt)
    torch.testing.assert_close(before[:, 2], after[:, 2], atol=1e-11, rtol=1e-11)


def test_cross_attention_uses_source_and_backpropagates_through_memory():
    model = small_model()
    tgt = torch.tensor([[1, 3]])
    a = model(torch.tensor([[3, 4, 2]]), tgt)
    b = model(torch.tensor([[5, 6, 7, 2]]), tgt)
    assert not torch.allclose(a, b)
    model_api.masked_cross_entropy(a, torch.tensor([[3, 2]]), 0).backward()
    assert model.src_embedding.weight.grad[3].abs().sum() > 0
    assert model.decoder[0].cross_attn.v_proj.weight.grad.abs().sum() > 0


def test_valid_integer_types_and_each_layer_trace_shapes():
    model = small_model(num_encoder_layers=2, num_decoder_layers=2)
    src = torch.tensor([[3, 2, 0]], dtype=torch.int16)
    tgt = torch.tensor([[1, 3]], dtype=torch.int8)
    logits, trace = model.forward_with_trace(src, tgt)
    assert logits.shape == (1, 2, 8) and logits.dtype == torch.float64
    for index in range(2):
        assert trace[f"encoder.{index}.output"].shape == (1, 3, 8)
        assert trace[f"decoder.{index}.output"].shape == (1, 2, 8)
    assert torch.equal(logits, model(src.long(), tgt.long()))


@pytest.mark.parametrize(
    "vocab,dtype", [(256, torch.uint8), (128, torch.int8), (32768, torch.int16)]
)
def test_narrow_integer_tokens_do_not_wrap_vocabulary_boundary(vocab, dtype):
    model = model_api.Transformer(
        model_api.TransformerConfig(vocab_size=vocab, d_model=2, num_heads=1, d_ff=4)
    )
    src = torch.tensor([[vocab - 1, 2]], dtype=dtype)
    tgt = torch.tensor([[1, vocab - 1]], dtype=dtype)
    labels = torch.tensor([[vocab - 1, 2]], dtype=dtype)
    logits = model(src, tgt)
    assert logits.shape == (1, 2, vocab)
    assert torch.isfinite(model_api.masked_cross_entropy(logits, labels, 0))
    torch.testing.assert_close(logits, model(src.long(), tgt.long()), rtol=0, atol=0)


def test_pad_outside_narrow_dtype_range_does_not_mask_real_zero_token():
    model = model_api.Transformer(
        model_api.TransformerConfig(
            vocab_size=257, pad_id=256, d_model=2, num_heads=1, d_ff=4
        )
    )
    src = torch.tensor([[0]], dtype=torch.uint8)
    tgt = torch.tensor([[0, 1]], dtype=torch.uint8)
    logits = model(src, tgt)
    torch.testing.assert_close(logits, model(src.long(), tgt.long()), rtol=0, atol=0)
    labels = torch.tensor([[0, 2]], dtype=torch.uint8)
    loss = model_api.masked_cross_entropy(
        torch.zeros(1, 2, 257).double(), labels, pad_id=256
    )
    assert loss.item() == pytest.approx(math.log(257), abs=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 3},
        {"d_model": 0},
        {"d_ff": -1},
        {"num_encoder_layers": 0},
        {"num_decoder_layers": 0},
        {"max_len": 0},
        {"pad_id": -1},
        {"pad_id": 16},
        {"eps": 0},
        {"eps": float("nan")},
        {"eps": float("inf")},
        {"num_heads": True},
        {"d_model": 8.0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((ValueError, TypeError)):
        model_api.TransformerConfig(**kwargs)


@pytest.mark.parametrize(
    "src,tgt",
    [
        ([[0, 0]], [[1, 3]]),
        ([[3, 2]], [[0, 1]]),
        ([[3, 8]], [[1, 3]]),
        ([[3, -1]], [[1, 3]]),
        ([[3, 2]], [[1, 8]]),
        ([[3, 2]], [[1, -1]]),
        ([[3, 2]], [[1.0, 3.0]]),
        ([[3.0, 2.0]], [[1, 3]]),
        ([[]], [[1, 3]]),
        ([[3, 2]], [[]]),
        ([[3, 2], [4, 2]], [[1, 3]]),
    ],
)
def test_forward_rejects_invalid_tokens(src, tgt):
    with pytest.raises((ValueError, TypeError)):
        small_model()(torch.tensor(src), torch.tensor(tgt))


def test_forward_rejects_sequence_longer_than_position_table():
    with pytest.raises(ValueError):
        small_model(max_len=2)(torch.tensor([[3, 4, 2]]), torch.tensor([[1, 3]]))


def test_masked_cross_entropy_matches_literal_and_ignores_pad():
    logits = torch.tensor([[[0.0, math.log(3)], [100.0, -100.0]]], requires_grad=True)
    loss = model_api.masked_cross_entropy(logits.double(), torch.tensor([[1, 0]]), 0)
    assert loss.item() == pytest.approx(math.log(4 / 3), abs=1e-7)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


def test_loss_rejects_all_pad_or_invalid_labels():
    logits = torch.randn(1, 2, 8, dtype=torch.float64)
    for labels in [
        torch.tensor([[0, 0]]),
        torch.tensor([[3, 8]]),
        torch.tensor([[3.5, 2]]),
    ]:
        with pytest.raises((ValueError, TypeError)):
            model_api.masked_cross_entropy(logits, labels, 0)


def test_full_model_gradient_matches_independent_finite_difference():
    model = small_model(d_model=4, num_heads=2, d_ff=7)
    src, tgt, labels = (
        torch.tensor([[3, 2, 0]]),
        torch.tensor([[1, 3]]),
        torch.tensor([[3, 2]]),
    )
    model_api.masked_cross_entropy(model(src, tgt), labels, 0).backward()
    parameter = model.encoder[0].self_attn.q_proj.weight
    analytic = parameter.grad[0, 1].item()
    old = parameter[0, 1].item()
    losses = []
    with torch.no_grad():
        for sign in [1, -1]:
            parameter[0, 1] = old + sign * 1e-5
            losses.append(model_api.masked_cross_entropy(model(src, tgt), labels, 0).item())
        parameter[0, 1] = old
    assert analytic == pytest.approx((losses[0] - losses[1]) / 2e-5, abs=2e-8, rel=1e-5)


def test_teacher_forcing_training_learns_copy_and_eos():
    model = small_model()
    src = torch.tensor([[3, 4, 2, 0], [5, 6, 2, 0]])
    tgt = torch.tensor([[1, 3, 4], [1, 5, 6]])
    labels = torch.tensor([[3, 4, 2], [5, 6, 2]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    initial = model_api.masked_cross_entropy(model(src, tgt), labels, 0).item()
    for _ in range(250):
        optimizer.zero_grad()
        loss = model_api.masked_cross_entropy(model(src, tgt), labels, 0)
        loss.backward()
        optimizer.step()
    final = model_api.masked_cross_entropy(model(src, tgt), labels, 0).item()
    assert final < initial * 0.1
    assert torch.equal(
        model.greedy_decode(src, 1, 2, 6), torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2]])
    )


def test_greedy_stops_at_eos_and_restores_training_mode():
    model = small_model()
    with torch.no_grad():
        model.generator.weight.zero_()
        model.generator.bias.fill_(-10)
        model.generator.bias[2] = 10
    assert model.training
    result = model.greedy_decode(torch.tensor([[3, 2], [4, 2]]), 1, 2, 7)
    assert torch.equal(result, torch.tensor([[1, 2], [1, 2]]))
    assert model.training
    assert torch.equal(
        model.greedy_decode(torch.tensor([[3, 2]]), 1, 2, 0), torch.tensor([[1]])
    )


def test_greedy_pads_finished_sample_while_other_sample_continues():
    model = small_model()
    src = torch.tensor([[3, 2, 0], [5, 6, 2]])
    tgt = torch.tensor([[1, 3, 0], [1, 5, 6]])
    labels = torch.tensor([[3, 2, 0], [5, 6, 2]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    for _ in range(250):
        optimizer.zero_grad()
        model_api.masked_cross_entropy(model(src, tgt), labels, 0).backward()
        optimizer.step()
    assert torch.equal(
        model.greedy_decode(src, 1, 2, 6), torch.tensor([[1, 3, 2, 0], [1, 5, 6, 2]])
    )


@pytest.mark.parametrize(
    "bos,eos,steps", [(0, 2, 2), (1, 0, 2), (1, 1, 2), (1, 8, 2), (1, 2, -1), (1, 2, 32)]
)
def test_greedy_rejects_invalid_special_tokens_and_lengths(bos, eos, steps):
    with pytest.raises((ValueError, TypeError)):
        small_model().greedy_decode(torch.tensor([[3, 2]]), bos, eos, steps)
