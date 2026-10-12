"""Final-hidden-state memory adapter. No global/request-shared generation cache."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class EvidenceAdapter(nn.Module):
    """target_mode="inferred" (v2) points at the target inside the facts; "explicit" is the v1 field reader."""
    def __init__(self, hidden_size, rank=128, generic=False, target_mode="inferred"):
        super().__init__()
        if target_mode not in ("inferred", "explicit"):
            raise ValueError("Unknown target mode")
        self.rank, self.generic, self.target_mode = rank, generic, target_mode
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.condition = nn.Linear(2 * rank, rank, bias=False)
        if target_mode == "explicit":
            self.missing_target = nn.Parameter(torch.zeros(rank))
        else:
            # Charge-conditioned pointer over fact tokens; generic shares it so only routing differs.
            self.target_e = nn.Linear(rank, rank, bias=False)
            self.target_c = nn.Linear(rank, rank, bias=False)
            self.target_score = nn.Linear(rank, 1)
        self.role_e = nn.Linear(rank, rank, bias=False)
        self.role_q = nn.Linear(rank, rank, bias=False)
        self.role = nn.Linear(rank, 3)
        # Generic control: 3 ordinary content heads; auxiliary roles never feed reading.
        heads = 3 if generic else 1
        self.query = nn.Linear(rank, heads * rank, bias=False)
        self.key = nn.Linear(rank, heads * rank, bias=False)
        self.value = nn.Linear(rank, rank, bias=False)
        self.gate = nn.Linear(2 * rank + 4, 1)
        self.up = nn.Linear(3 * rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def project(self, h):
        return self.down(self.norm(h.to(self.norm.weight.dtype)))

    def memory(self, hidden, meta):
        if hidden.ndim != 2 or not 0 < meta["prompt_length"] <= len(hidden):
            raise ValueError("Invalid per-request prompt boundary")
        def pool(indices):
            if not indices or min(indices) < 0 or max(indices) >= meta["prompt_length"]:
                raise ValueError("Memory must contain prompt tokens only")
            return hidden[indices].mean(0)
        e = self.project(torch.stack([pool(s) for s in meta["sentences"]]))
        charge = self.project(pool(meta["charge"]))
        extra = {}
        if self.target_mode == "explicit":
            target = self.project(pool(meta["target"])) if meta["target"] else self.missing_target
        else:
            if meta["target"]:
                raise ValueError("Latent-target TERA never reads an explicit target field")
            facts = sorted({i for s in meta["sentences"] for i in s})
            z = self.project(hidden[facts])
            scores = self.target_score(torch.tanh(self.target_e(z) + self.target_c(charge))).squeeze(-1)
            pointer = scores.float().softmax(0)
            target = (pointer[:, None] * z.float()).sum(0).to(z.dtype)
            extra["pointer"] = pointer
        q = self.condition(torch.cat((target, charge)))
        role_logits = self.role(torch.tanh(self.role_e(e) + self.role_q(q)))
        return {"q": q, "roles": role_logits, "key": self.key(e), "value": self.value(e)} | extra

    @staticmethod
    def pointer_loss(memory, meta):
        """-log pointer mass on literal target mentions; None when unsupervised."""
        mentions = meta.get("target_mentions")
        if "pointer" not in memory or not mentions:
            return None
        facts = sorted({i for s in meta["sentences"] for i in s})
        index = {position: k for k, position in enumerate(facts)}
        if any(m not in index for m in mentions):
            raise ValueError("Target mention outside fact memory")
        mass = memory["pointer"][[index[m] for m in mentions]].sum()
        return -mass.clamp_min(1e-8).log()

    def forward(self, hidden, memory):
        z, q = self.project(hidden), memory["q"]
        queries, keys = self.query(z), memory["key"]
        if self.generic:
            attn = torch.einsum("thr,nhr->thn", queries.reshape(-1, 3, self.rank), keys.reshape(-1, 3, self.rank))
            attn = (attn.float() / math.sqrt(self.rank)).softmax(-1)
            values = torch.einsum("thn,nr->thr", attn, memory["value"].float())
            mass = attn.amax(-1)
            entropy = -(attn * attn.clamp_min(1e-8).log()).sum(-1).mean(-1, keepdim=True)
        else:
            attn = (queries.float() @ keys.float().T / math.sqrt(self.rank)).softmax(-1)
            roles = memory["roles"].float().softmax(-1)
            routed = attn[:, :, None] * roles[None, :, :]
            values = torch.einsum("tnc,nr->tcr", routed, memory["value"].float())
            mass = routed.sum(1)
            role_entropy = -(roles * roles.clamp_min(1e-8).log()).sum(-1)
            entropy = (attn @ role_entropy)[:, None]
        gate_input = torch.cat((z.float(), q.float().expand(len(z), -1), mass, entropy), -1)
        gate = self.gate(gate_input).sigmoid()
        delta = self.up(values.flatten(1)) * gate
        return hidden + delta.to(hidden.dtype)


class MemoryCausalLM(nn.Module):
    """Batch size one by design; batching would need independent span/caching metadata."""
    def __init__(self, backbone, arm, rank=128, target_mode="inferred"):
        super().__init__()
        if arm not in ("direct", "generic", "tera_noaux", "tera"):
            raise ValueError("Unknown module arm")
        self.backbone, self.arm = backbone, arm
        self.config = self.base.config
        self.adapter = None if arm == "direct" else EvidenceAdapter(self.config.hidden_size, rank, arm == "generic", target_mode)
        self.aux_weight = 0.1 if arm in ("generic", "tera") else 0.0
        self.last_losses = {}

    @property
    def base(self):
        return self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        # Preserve the installed Transformers contract, including layer stride/offload.
        return self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs)

    def forward(self, input_ids, labels, meta, role_labels, sentence_mask=None):
        if input_ids.shape[0] != 1 or input_ids.shape != labels.shape:
            raise ValueError("TERA v1 requires unpadded batch size one")
        p = meta["prompt_length"]
        if not 0 < p < input_ids.shape[1] or (labels[0, :p] != -100).any() or (labels[0, p:] == -100).any():
            raise ValueError("Invalid completion-only loss boundary")
        hidden = self.base.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[0]
        # Shift once: position P-1 predicts first assistant token; no earlier update.
        predicted = hidden[p-1:-1]
        auxiliary = hidden.new_zeros((), dtype=torch.float32)
        pointer_loss = None
        if self.adapter is not None:
            memory = self.adapter.memory(hidden, meta)
            predicted = self.adapter(predicted, memory)
            roles = torch.as_tensor(role_labels, device=hidden.device)
            if len(roles) != len(memory["roles"]) or not torch.isin(roles, roles.new_tensor([-100, 0, 1, 2])).all():
                raise ValueError("Invalid auxiliary labels")
            if (roles != -100).any():
                auxiliary = F.cross_entropy(memory["roles"].float(), roles, ignore_index=-100)
            pointer_loss = self.adapter.pointer_loss(memory, meta)
            if pointer_loss is not None:
                auxiliary = auxiliary + pointer_loss
        # Only completion logits are needed, avoiding a full prompt x vocabulary tensor.
        logits = self.base.lm_head(predicted)
        if sentence_mask is None:
            sft = F.cross_entropy(logits.float(), labels[0, p:])
            sentence_loss = rest_loss = None
        else:
            mask = torch.as_tensor(sentence_mask, device=labels.device)
            if mask.dtype != torch.bool or mask.shape != labels.shape or mask[0, :p].any():
                raise ValueError("Invalid sentence loss mask")
            mask = mask[0, p:]
            if not mask.any() or mask.all():
                raise ValueError("Sentence and remaining completion both need supervised tokens")
            per_token = F.cross_entropy(logits.float(), labels[0, p:], reduction="none")
            sentence_loss, rest_loss = per_token[mask].mean(), per_token[~mask].mean()
            sft = .5 * sentence_loss + .5 * rest_loss
        loss = sft + self.aux_weight * auxiliary
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        self.last_losses = {"sft": float(sft.detach()), "auxiliary": float(auxiliary.detach())}
        if pointer_loss is not None:
            self.last_losses["pointer"] = float(pointer_loss.detach())
        if sentence_loss is not None:
            self.last_losses.update(sentence=float(sentence_loss.detach()), rest=float(rest_loss.detach()))
        return {"loss": loss}

    def step(self, input_ids, meta=None, memory=None, past=None, return_base_logits=False):
        if input_ids.shape[0] != 1:
            raise ValueError("TERA v1 decodes one independent request at a time")
        output = self.base.model(input_ids=input_ids, past_key_values=past, use_cache=True, return_dict=True)
        hidden = output.last_hidden_state[0]
        base_logits = self.base.lm_head(hidden[-1:])[0].float() if return_base_logits else None
        if self.adapter is not None:
            if past is None:
                if meta is None or meta["prompt_length"] != len(hidden) or memory is not None:
                    raise ValueError("Prefill needs its own complete prompt, not another request's memory")
                memory = self.adapter.memory(hidden, meta)
            elif memory is None:
                raise ValueError("Missing request-local memory")
            hidden = self.adapter(hidden[-1:], memory)
        logits = self.base.lm_head(hidden[-1:])[0].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite generation logits")
        result = (logits, memory, output.past_key_values)
        return result + (base_logits,) if return_base_logits else result
