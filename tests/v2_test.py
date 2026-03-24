import time
import torch
import sys
from multivariategpt.multivariategpt_v2 import GPT as GPT_v2, GPTConfig as GPTConfig_v2
from multivariategpt.multivariategpt import GPT as GPT_v1, GPTConfig as GPTConfig_v1

# tests to make sure that the v2 matches v1 numerically but is faster

cfg = GPTConfig_v2(batch_size=2, block_size=8, n_embd=32, n_heads=2, n_layer=2, vocab_size=16, n_head_blocks=0)
model = GPT_v2(cfg)

B, T = 2, 8
c  = torch.randint(0, 16, (B, T))
v  = torch.randn(B, T)
tc = torch.randint(0, 16, (B, T))
tv = torch.randn(B, T)

# Training pass: x_v_l and x_v_s should be [B,T] loc/scale for the true next token
xc, xvl, xvs, cl, vl, loss = model(c, v, tc, tv)
assert xvl.shape == (B, T) and xvs.shape == (B, T), "expected [B,T] loc/scale during training"
assert loss is not None
print(f"[n_head_blocks=0] training loss={loss.item():.4f}, c_loss={cl.item():.4f}, v_loss={vl.item():.4f}")

# Inference pass: full tensors must exist
xc, xvl, xvs, cl, vl, loss = model(c, v)
assert xvl is not None and xvl.shape == (B, T, 16)
assert loss is None
print(f"[n_head_blocks=0] inference x_v_l shape={xvl.shape} OK")

# generate() test
c_g, v_g = model.generate(c[:1], v[:1], max_new_tokens=3)
assert c_g.shape == (1, T + 3)
print(f"[n_head_blocks=0] generate output shape={c_g.shape} OK")

# n_head_blocks=1
cfg2 = GPTConfig_v2(batch_size=2, block_size=8, n_embd=32, n_heads=2, n_layer=2, vocab_size=16, n_head_blocks=1)
model2 = GPT_v2(cfg2)
xc2, xvl2, xvs2, cl2, vl2, loss2 = model2(c, v, tc, tv)
assert xvl2.shape == (B, T) and loss2 is not None
print(f"[n_head_blocks=1] training loss={loss2.item():.4f} OK")

# backward through efficient path
loss2.backward()
assert model2.v_head_l.weight.grad is not None
print(f"[n_head_blocks=1] v_head_l.weight.grad norm={model2.v_head_l.weight.grad.norm().item():.4f} OK")

# default backward-compat
assert GPTConfig_v2().n_head_blocks == 0
print(f"[default] n_head_blocks=0 OK")

# v1 / v2 numeric equivalence (n_head_blocks=0 with shared random weights)
# Both models should produce identical loss when given the same weights and inputs.
torch.manual_seed(0)
cfg_eq = dict(block_size=18, n_embd=32, n_heads=2, n_layer=12, vocab_size=16)
model_eq_v1 = GPT_v1(GPTConfig_v1(**cfg_eq))
model_eq_v2 = GPT_v2(GPTConfig_v2(**cfg_eq, n_head_blocks=0))
# copy weights from v1 into v2
model_eq_v2.load_state_dict(model_eq_v1.state_dict())

torch.manual_seed(1)
c_eq  = torch.randint(0, 16, (2, 8))
v_eq  = torch.randn(2, 8)
tc_eq = torch.randint(0, 16, (2, 8))
tv_eq = torch.randn(2, 8)

with torch.no_grad():
    _, _, _, _, _, loss_v1 = model_eq_v1(c_eq, v_eq, tc_eq, tv_eq)
    _, _, _, _, _, loss_v2 = model_eq_v2(c_eq, v_eq, tc_eq, tv_eq)

assert torch.allclose(loss_v1, loss_v2, atol=1e-5), \
    f"v1/v2 loss mismatch: v1={loss_v1.item():.8f}  v2={loss_v2.item():.8f}"
print(f"[numeric equivalence] v1={loss_v1.item():.8f}  v2={loss_v2.item():.8f}  OK")

print("\nAll checks passed.")


# ── Performance test ─────────────────────────────────────────────────────────
# Full forward pass v1 (materializes [B,T,vocab_size] value tensors)
# vs v2 (efficient gather path). Large vocab to amplify the difference.

print("\n--- Performance test ---")

WARMUP = 10
REPS   = 100

torch.manual_seed(42)
perf_cfg = dict(
    vocab_size=16384, n_embd=256, n_heads=4, n_layer=4,
    block_size=32, dropout=0, bias=0, n_head_blocks=0,
)

from multivariategpt.multivariategpt_v2 import GPT as GPT_v2, GPTConfig as GPTConfig_v2
from multivariategpt.multivariategpt import GPT as GPT_v1, GPTConfig as GPTConfig_v1

model_v2 = GPT_v2(GPTConfig_v2(**perf_cfg))
model_v1 = GPT_v1(GPTConfig_v1(**{k: v for k, v in perf_cfg.items() if k != 'n_head_blocks'}))
model_v2.eval()
model_v1.eval()

B_p, T_p, C_p = 2, 32, 16384
c_p  = torch.randint(0, C_p, (B_p, T_p))
v_p  = torch.randn(B_p, T_p)
tc_p = torch.randint(0, C_p, (B_p, T_p))
tv_p = torch.randn(B_p, T_p)

with torch.no_grad():
    for _ in range(WARMUP):
        model_v1(c_p, v_p, tc_p, tv_p)
        model_v2(c_p, v_p, tc_p, tv_p)

with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(REPS):
        model_v1(c_p, v_p, tc_p, tv_p)
    t_v1 = (time.perf_counter() - t0) / REPS * 1000

with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(REPS):
        model_v2(c_p, v_p, tc_p, tv_p)
    t_v2 = (time.perf_counter() - t0) / REPS * 1000

speedup = t_v1 / t_v2
print(
    f"forward pass timing  v1={t_v1:.3f}ms  v2={t_v2:.3f}ms  "
    f"speedup={speedup:.2f}x  (vocab={C_p}, n_embd=256, B={B_p}, T={T_p})"
)
assert speedup > 1.0, f"Expected v2 to be faster, got speedup={speedup:.2f}x"
print("PASS test_value_head_speedup")

