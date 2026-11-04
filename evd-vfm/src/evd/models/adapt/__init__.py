"""PEFT adaptation strategies, the core of contribution.

``base`` is the contract shared by all families ``target_maps`` says where to adapt and the other 
modules say how:

* ``reparam``: LoRA, rsLoRA and DoRA, with the ``separate``, ``fused``, ``fused_nr``
  and ``shared_b`` factorizations plus the ``qk_only`` placement;
* ``prompt``: shallow and deep VPT, and the sequential bank machinery;
* ``apt``: prompts as keys and values inside attention;
* ``davpt``: deep VPT whose last banks are class proxies in a metric-learning loss;
* ``prompt_family``: the factory choosing among the prompt strategies;
* ``composite``: two families on the same backbone.

``reparam`` comes from the verified prototype ``../lab/scripts/s3_target_modules.py``
"""

from evd.models.adapt.apt import (
    KV_PROMPT_ATTENTION,
    AptPEFT,
    KVPromptedAttention,
    KVPromptTokens,
    apt_overhead,
)
from evd.models.adapt.base import (
    ADAPTERS,
    AdaptationReport,
    AdaptationStrategy,
    AdaptedBackbone,
    AuxiliaryLoss,
    Family,
    NoAdapt,
    freezes_backbone,
    preserves_backbone_output,
)
from evd.models.adapt.composite import (
    CANONICAL_ORDER,
    CompositeAdapt,
)
from evd.models.adapt.davpt import (
    DaVptPEFT,
    ProxyCriterion,
    ProxyLossSpec,
    ProxyPromptLoss,
    masked_logsumexp,
)
from evd.models.adapt.factory import (
    ADAPT_BUILDERS,
    AdaptContext,
    build_adaptation,
)
from evd.models.adapt.prompt import (
    ALLOWED_POLICIES,
    PROMPT_VARIANTS,
    SEQUENTIAL_VARIANTS,
    PosPolicy,
    PromptedBlock,
    PromptNorm,
    PromptPEFT,
    PromptSpec,
    PromptTokens,
    PromptVariant,
    prompt_overhead,
    tubelet_numel,
    validate_pos_policy,
    xavier_uniform_bound,
)
from evd.models.adapt.prompt_family import make_prompt_strategy
from evd.models.adapt.reparam import (
    FACTORIZATIONS,
    AdaptedLinear,
    DoRABlock,
    DoRALinear,
    Factorization,
    FusedQKVLoRALinear,
    LoRABlock,
    LoRALinear,
    LowRankBlock,
    ReparamPEFT,
    ReparamSpec,
    SharedBBlock,
    build_blocks,
    effective_rank,
    lora_scaling,
    trainable_parameters_formula,
)
from evd.models.adapt.target_maps import (
    OutSlice,
    Placement,
    Role,
    TargetModuleMap,
    TargetRef,
)

__all__ = [
    "ADAPTERS",
    "ADAPT_BUILDERS",
    "FACTORIZATIONS",
    "KV_PROMPT_ATTENTION",
    "CANONICAL_ORDER",
    "ALLOWED_POLICIES",
    "PROMPT_VARIANTS",
    "SEQUENTIAL_VARIANTS",
    "AdaptContext",
    "AdaptationReport",
    "AdaptationStrategy",
    "AdaptedBackbone",
    "AdaptedLinear",
    "AptPEFT",
    "AuxiliaryLoss",
    "CompositeAdapt",
    "DaVptPEFT",
    "DoRABlock",
    "DoRALinear",
    "Factorization",
    "Family",
    "FusedQKVLoRALinear",
    "KVPromptTokens",
    "KVPromptedAttention",
    "LoRABlock",
    "LoRALinear",
    "LowRankBlock",
    "NoAdapt",
    "OutSlice",
    "Placement",
    "PosPolicy",
    "PromptNorm",
    "PromptPEFT",
    "PromptSpec",
    "PromptTokens",
    "PromptVariant",
    "PromptedBlock",
    "ProxyCriterion",
    "ProxyLossSpec",
    "ProxyPromptLoss",
    "ReparamPEFT",
    "ReparamSpec",
    "Role",
    "SharedBBlock",
    "TargetModuleMap",
    "TargetRef",
    "apt_overhead",
    "build_adaptation",
    "build_blocks",
    "effective_rank",
    "freezes_backbone",
    "lora_scaling",
    "preserves_backbone_output",
    "make_prompt_strategy",
    "masked_logsumexp",
    "prompt_overhead",
    "trainable_parameters_formula",
    "tubelet_numel",
    "validate_pos_policy",
    "xavier_uniform_bound",
]
