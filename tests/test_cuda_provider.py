"""The emitted graph on ONNX Runtime's CUDA executor (§spec:verification).

The CPU executor is where the suite verifies every graph, and a consumer runs
the graph on a GPU. These tests hold the same lattice against the same CPU
reference with the graph on the device, at `cuda.GPU_TOLERANCE`, because GPU
transcendentals are not libm whichever executor issues them.

Every node runs on the device or the session fails (`oracle.run_graph`), so a
graph that agrees here agreed on the GPU. TF32 is off: ONNX Runtime enables
it by default and it rounds the 1x1 convolution's inputs to a 10-bit
mantissa, which is a choice a consumer makes about precision, not a property
of the graph.

The tests skip wherever the CUDA build of ONNX Runtime or a device is absent,
which is every hosted runner; the bench GPU workflow runs them.
"""

import PyOpenColorIO as OCIO
import pytest

from ocio_codegen.addressing import DEFAULT_CONFIG, Resolved, enumerate_transforms
from ocio_codegen.compiler import compile_processor
from ocio_codegen.cuda import GPU_TOLERANCE
from ocio_codegen.oracle import compare, cpu_reference, lattice, run_graph

CUDA_PROVIDER = "CUDAExecutionProvider"

#: Full float32 arithmetic in the convolution, as the CPU executor computes it.
FP32 = {"use_tf32": "0"}


def on_device(processor, model, parameters=None):
    """Hold one graph, run on the device, against the CPU reference."""
    samples = lattice(processor)
    return compare(
        cpu_reference(processor, samples),
        run_graph(
            model,
            samples,
            parameters,
            provider=CUDA_PROVIDER,
            provider_options=FP32,
        ),
        samples,
        tolerance=GPU_TOLERANCE,
    )


@pytest.fixture(scope="module", autouse=True)
def cuda_provider():
    """Skip where the CUDA executor is not installed or reaches no device, so
    these tests fail only on answers, never on machinery. The workflow probes
    the same machinery first and fails there instead of skipping."""
    ort = pytest.importorskip("onnxruntime")
    if CUDA_PROVIDER not in ort.get_available_providers():
        pytest.skip(f"this onnxruntime build carries no {CUDA_PROVIDER}")
    # The CUDA and cuDNN libraries come from the gpu extra's NVIDIA wheels,
    # which are not on the loader's search path until preloaded.
    ort.preload_dlls()


def test_every_graph_agrees_on_the_device(config):
    """The pinned config's 159 transforms, each compiled once and run on the
    GPU. A session the executor refuses is reported by transform, beside the
    disagreements, so one run names everything that does not hold."""
    refused = []
    disagreements = []
    for label, processor in enumerate_transforms(config, uri=DEFAULT_CONFIG):
        model = compile_processor(
            Resolved(
                processor=processor,
                config_name="device",
                config_uri=DEFAULT_CONFIG,
                endpoints=label,
            )
        )
        try:
            result = on_device(processor, model)
        except Exception as exc:  # ONNX Runtime raises its own unexported types
            refused.append((label, str(exc).splitlines()[0]))
            continue
        if not result.ok:
            disagreements.append((label, str(result)))
    assert refused == []
    assert disagreements == []


def test_a_live_parameter_binds_on_the_device(config, compile_bare):
    """A graph input is bound per run (§spec:dynamic-properties), and the
    value reaches the device: one graph, swept, agrees with OCIO built at the
    swept value."""
    dynamic = OCIO.ExposureContrastTransform(style=OCIO.EXPOSURE_CONTRAST_LINEAR)
    dynamic.makeExposureDynamic()
    model = compile_bare(config.getProcessor(dynamic))

    swept = 2.5
    fixed = OCIO.ExposureContrastTransform(
        style=OCIO.EXPOSURE_CONTRAST_LINEAR, exposure=swept
    )
    result = on_device(
        config.getProcessor(fixed), model, parameters={"EXPOSURE": [swept]}
    )
    assert result.ok, str(result)
