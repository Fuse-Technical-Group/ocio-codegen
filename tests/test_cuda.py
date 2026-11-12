"""The fused CUDA kernel artifact (§spec:cuda-kernel).

The ONNX graph is the portable artifact; the kernel exists for the one
transform class that defeats graph executors — the ACES 2.0 display renders,
whose fusion-hostile op chain costs a memory pass per segment under TensorRT.
The kernel is transpiled from the shader OCIO's own GPU renderer emits, so
the tests hold three claims: the source is self-contained, the transpiler
refuses what it cannot carry, and the compiled kernel agrees with the same
oracle every graph answers to — at the GPU tolerance, which is the band
OCIO's own GPU renderer occupies against its CPU renderer.
"""

import re
import sys

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio_codegen import cuda
from ocio_codegen.addressing import (
    DEFAULT_CONFIG,
    OPTIMIZATION_FLAGS,
    Resolved,
    enumerate_transforms,
    reference_space,
    resolve_colorspaces,
    resolve_display_view,
)

#: The transform the kernel exists for: the heaviest display render in the
#: pinned config, and the one §spec:cuda-kernel names.
DISPLAY = "sRGB - Display"
ACES2_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"

#: A closed-form display render, so the transpiler is held on a shader with
#: no textures at all.
PLAIN_VIEW = "Un-tone-mapped"


#: The architecture the census compiles for: the oldest NVRTC 13 targets, so
#: a kernel that compiles here compiles for every device the toolkit serves.
#: NVRTC needs no device to compile, so the census runs in CPU-only CI.
CENSUS_ARCH = "compute_75"

#: The fused-kernel partition of the pinned config's 159 transforms
#: (§spec:cuda-kernel). ``refused`` carries a ``Lut1D`` whose table OCIO
#: publishes as a two-dimensional texture, which the transpiler names and the
#: graph serves; ``compiled`` is every other transform, through NVRTC.
#: Pinned the way ``test_census.OP_CENSUS`` is: a builtin form the prelude
#: lacks moves a transform out of ``compiled``.
KERNEL_CENSUS = {"compiled": 119, "refused": 40}

#: Every closed-form curve family the pinned config carries, each named by
#: one color space: the camera log curves (``LogCamera``, ``Log``) and the
#: display gammas (``ExponentWithLinear``, ``Exponent``). Their shaders mix
#: vector and scalar builtin arguments that the ACES chain does not.
CURVES = [
    "Log3G10 REDWideGamutRGB",
    "ACEScct",
    "ARRI LogC3 (EI800)",
    "ARRI LogC4",
    "BMDFilm WideGamut Gen5",
    "DaVinci Intermediate WideGamut",
    "D-Log D-Gamut",
    "V-Log V-Gamut",
    "S-Log3 S-Gamut3",
    "sRGB Encoded Rec.709 (sRGB)",
    "Gamma 2.4 Encoded Rec.709",
    "Camera Rec.709",
]

UNARY_BUILTINS = (
    "abs",
    "sign",
    "floor",
    "ceil",
    "sqrt",
    "exp",
    "exp2",
    "log",
    "log2",
    "sin",
    "cos",
    "atan",
)

#: One call per GLSL builtin form OCIO's GPU emitters write, over every
#: vector width: componentwise math, the mixed vector/scalar forms GLSL 4.0
#: defines for ``min``/``max``/``clamp``/``mix``/``step``, the scalar-first
#: ``max(0.01, v)`` spelling OCIO emits, and the comparisons it wraps in a
#: vector constructor. ``V`` stands for the width under test.
BUILTIN_FORMS = [
    *(f"{name}(v)" for name in UNARY_BUILTINS),
    "pow(v, v)",
    "atan(v, v)",
    "min(v, v)",
    "min(v, s)",
    "min(s, v)",
    "max(v, v)",
    "max(v, s)",
    "max(s, v)",
    "step(v, v)",
    "step(s, v)",
    "clamp(v, v, v)",
    "clamp(v, s, s)",
    "mix(v, v, v)",
    "mix(v, v, s)",
    "V(greaterThan(v, v))",
    "V(lessThan(v, v))",
    "V(greaterThanEqual(v, v))",
    "V(lessThanEqual(v, v))",
    "V(any(greaterThan(v, v)) ? s : dot(v, v))",
    "v + v * v - v / v",
    "s + v * s - v / s",
    "s * v - s / v",
    "-v",
    "V(s)",
]


def compiles(source):
    """NVRTC's verdict on a source, at the census architecture, on no device."""
    cuda.compile_ptx(source, CENSUS_ARCH)


def kernel_census(config):
    """Each transform in the pinned config with its kernel source, or with
    the transpiler's refusal in place of one."""
    for label, processor in enumerate_transforms(config, uri=DEFAULT_CONFIG):
        resolved = Resolved(
            processor=processor,
            config_name="census",
            config_uri=DEFAULT_CONFIG,
            endpoints=label,
        )
        try:
            source = cuda.kernel_source(resolved)
        except cuda.UnsupportedShaderError as refusal:
            yield label, resolved, None, refusal
        else:
            yield label, resolved, source, None


@pytest.fixture(scope="module")
def nvrtc():
    """NVRTC, which the dev extra installs wherever NVIDIA publishes it.

    Absent only on macOS, where no wheel exists; anywhere else a missing
    library is a broken environment, and the census fails rather than skips.
    """
    if sys.platform == "darwin":
        pytest.importorskip("cuda.bindings.nvrtc")
    from cuda.bindings import nvrtc

    return nvrtc


@pytest.fixture(scope="module")
def aces2_sdr(config, config_uri):
    return resolve_display_view(
        config, DISPLAY, ACES2_VIEW, src="ACEScg", uri=config_uri
    )


@pytest.fixture(scope="module")
def untonemapped(config, config_uri):
    return resolve_display_view(
        config, DISPLAY, PLAIN_VIEW, src="ACEScg", uri=config_uri
    )


@pytest.fixture(scope="module")
def red_log_camera(config, config_uri):
    """A camera log source into the ACES 2.0 render: the shader's LogCamera
    segment compares three-component vectors, where the ACES chain alone
    compares only four."""
    return resolve_display_view(
        config, DISPLAY, ACES2_VIEW, src="Log3G10 REDWideGamutRGB", uri=config_uri
    )


@pytest.fixture(scope="module")
def source(aces2_sdr):
    return cuda.kernel_source(aces2_sdr)


def bare(config, config_uri, transform, label="bare"):
    """A hand-built transform resolved the way ``compile_bare`` does."""
    return Resolved(
        processor=config.getProcessor(transform).getOptimizedProcessor(
            OPTIMIZATION_FLAGS
        ),
        config_name="bare",
        config_uri=config_uri,
        endpoints=label,
    )


class TestSource:
    def test_self_contained(self, source):
        """No include, no sampler, no GL call survives: a consumer compiles
        the file with NVRTC and nothing else."""
        assert "#include" not in source
        assert "uniform" not in source
        assert "texture(" not in source
        assert 'extern "C"' in source

    def test_both_entry_points(self, source):
        """Planar RGB at float32 and float16 (§spec:cuda-kernel). Alpha is
        absent, not passed through (§spec:emitted-graph)."""
        assert "apply_f32" in source
        assert "apply_f16" in source
        assert "alpha" not in source.lower()

    def test_tables_are_embedded(self, source):
        """Both published textures land as global arrays — global rather than
        constant memory, which serializes on divergent indices."""
        assert "__device__ const float ocio_reach_m_table_0_data[363]" in source
        assert "__device__ const float ocio_gamut_cusp_table_0_data[1089]" in source
        assert "__constant__" not in source

    def test_metadata_header(self, source, config_uri):
        """The artifact says what produced it, like the graph does."""
        assert config_uri in source
        assert f"{DISPLAY} / {ACES2_VIEW}" in source

    def test_float_literals_are_float(self, source):
        """An unsuffixed literal is a double, and one double in a chain drops
        the whole expression to double-precision arithmetic."""
        unsuffixed = re.compile(
            r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
        )
        for line in source.splitlines():
            code = line.split("//")[0]
            assert not unsuffixed.search(code), line

    def test_closed_form_shader_transpiles(self, untonemapped):
        """A shader with no tables at all comes through the same path."""
        source = cuda.kernel_source(untonemapped)
        assert 'extern "C"' in source
        assert "apply_f32" in source


class TestRefusal:
    def test_dynamic_property_refused(self, config, config_uri):
        """A dynamic property is a uniform, and the kernel bakes everything;
        the refusal names the mechanism rather than emitting a stale value."""
        transform = OCIO.ExposureContrastTransform()
        transform.makeExposureDynamic()
        with pytest.raises(cuda.UnsupportedShaderError, match="uniform"):
            cuda.kernel_source(bare(config, config_uri, transform))

    def test_interpolated_texture_refused(self, config, config_uri):
        """A linearly-interpolated texture has no transpiled equivalent yet;
        the refusal names the texture. The table is non-identity because an
        identity ``Lut1D`` optimizes away before any texture is published."""
        transform = OCIO.Lut1DTransform(length=8)
        for i in range(8):
            value = (i / 7) ** 0.5
            transform.setValue(i, value, value, value)
        with pytest.raises(cuda.UnsupportedShaderError, match="ocio_lut1d"):
            cuda.kernel_source(bare(config, config_uri, transform))


@pytest.mark.usefixtures("nvrtc")
class TestCompile:
    """Every kernel the transpiler emits compiles. NVRTC compiles for a named
    architecture without a device, so this holds in CPU-only CI."""

    @pytest.mark.parametrize("width", ["vec2", "vec3", "vec4"])
    @pytest.mark.parametrize("form", BUILTIN_FORMS)
    def test_prelude_carries_builtin_form(self, form, width):
        expression = form.replace("V(", f"{width}(")
        compiles(
            cuda._PRELUDE
            + f"__device__ {width} probe({width} v, float s) {{\n"
            + f"    {width} r = v;\n"
            + f"    r = {width}({expression});\n"
            + "    r += v; r -= v; r *= v; r /= v; r *= s; r /= s;\n"
            + "    return r;\n}\n"
        )

    def test_every_transform_compiles_or_is_refused(self, config):
        """The census the specification quotes. A transpiled kernel NVRTC
        rejects is a failure naming the transform and NVRTC's first error."""
        census = {"compiled": 0, "refused": 0}
        failures = []
        for label, _, source, refusal in kernel_census(config):
            if refusal is not None:
                assert "ocio_lut1d" in str(refusal), label
                census["refused"] += 1
                continue
            try:
                compiles(source)
            except RuntimeError as exc:
                errors = re.findall(r"error: (.*)", str(exc))
                failures.append((label, errors[:1]))
                continue
            census["compiled"] += 1
        assert failures == []
        assert census == KERNEL_CENSUS


@pytest.fixture(scope="module")
def cuda_runtime():
    """Skip where the NVRTC library or a CUDA device is absent, so the
    execution tests fail only on answers, never on machinery."""
    pytest.importorskip("cuda.bindings")
    try:
        cuda.device_arch()
    except Exception as exc:  # any miss means "no GPU here", not a failure
        pytest.skip(f"no usable CUDA runtime: {exc}")


@pytest.mark.usefixtures("cuda_runtime")
class TestKernel:
    def test_kernel_agrees_with_oracle(self, aces2_sdr):
        result = cuda.verify(aces2_sdr)
        assert result.ok, str(result)

    def test_closed_form_kernel_agrees(self, untonemapped):
        result = cuda.verify(untonemapped)
        assert result.ok, str(result)

    def test_camera_log_source_compiles_and_agrees(self, red_log_camera):
        """A camera log curve's shader compares vec3s; the kernel compiles and
        agrees with the oracle from that source too."""
        result = cuda.verify(red_log_camera)
        assert result.ok, str(result)

    @pytest.mark.parametrize("direction", ["decode", "encode"])
    @pytest.mark.parametrize("space", CURVES)
    def test_curve_agrees(self, config, config_uri, space, direction):
        """Each curve family both ways against the reference. The encode
        direction takes vector ``log`` and scalar-first ``max``."""
        reference = reference_space(config)
        src, dst = (space, reference) if direction == "decode" else (reference, space)
        result = cuda.verify(resolve_colorspaces(config, src, dst, uri=config_uri))
        assert result.ok, str(result)

    def test_every_compiled_transform_agrees(self, config):
        """The census, executed: every transform the transpiler accepts
        agrees with the oracle at `GPU_TOLERANCE`."""
        disagreements = []
        for label, resolved, source, refusal in kernel_census(config):
            if refusal is not None:
                continue
            result = cuda.verify(resolved, source)
            if not result.ok:
                disagreements.append((label, str(result)))
        assert disagreements == []

    def test_f16_entry_point_tracks_f32(self, aces2_sdr, source):
        """The half kernel is the same arithmetic behind quantized edges, so
        it agrees with the float kernel to half precision, not to the oracle
        tolerance (§spec:cuda-kernel)."""
        from ocio_codegen.oracle import lattice

        samples = lattice(aces2_sdr.processor)
        f32 = cuda.run_kernel(source, samples)
        f16 = cuda.run_kernel(source, samples, precision="f16")
        assert np.isfinite(f16).all()
        assert np.abs(f16 - f32).max() < 1e-2
