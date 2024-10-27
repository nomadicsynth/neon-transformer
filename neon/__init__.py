# Copyright 2024 Neon Cortex and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {
    "configuration_neon": ["NeonConfig"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_neon"] = [
        "NeonForCausalLM",
        "NeonModel",
        "NeonPreTrainedModel",
        "NeonForSequenceClassification",
        "NeonForTokenClassification",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_neon"] = [
        "FlaxNeonForCausalLM",
        "FlaxNeonModel",
        "FlaxNeonPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_neon"] = [
        "TFNeonModel",
        "TFNeonForCausalLM",
        "TFNeonForSequenceClassification",
        "TFNeonPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_neon import NeonConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_neon import (
            NeonForCausalLM,
            NeonForSequenceClassification,
            NeonForTokenClassification,
            NeonModel,
            NeonPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_neon import (
            FlaxNeonForCausalLM,
            FlaxNeonModel,
            FlaxNeonPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_neon import (
            TFNeonForCausalLM,
            TFNeonForSequenceClassification,
            TFNeonModel,
            TFNeonPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
