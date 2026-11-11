"""A display render led by a live ASC CDL (§spec:graded-render).

A camera's look travels beside its pictures as ten numbers that change while
the show runs, so the grade cannot be part of the processor OCIO resolves: at
its identity OCIO folds it away, and at any other value it bakes the numbers
in. The request names the grade's style instead, and both artifacts lead with
the grade as live parameters — graph inputs for the graph, a by-value argument
for the kernel. Each is held against OCIO's own CPU processor for the same CDL
in front of the same render, at several grades, so the claim is about one
artifact across the knob's range rather than one setting of it.
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio_codegen import compile_display_view, cuda
from ocio_codegen.addressing import (
    GRADE_KEY,
    IDENTITY_GRADE,
    AddressError,
    Grade,
    resolve_display_view,
)
from ocio_codegen.builder import parameters
from ocio_codegen.compiler import compile_processor, op_names
from ocio_codegen.emitters import (
    CDL_OFFSET,
    CDL_POWER,
    CDL_SATURATION,
    CDL_SLOPE,
)
from ocio_codegen.oracle import verify

#: A camera's log encoding into the ACES 2.0 SDR render: the transform a graded
#: camera feed is put on a monitor with.
SOURCE = "Log3G10 REDWideGamutRGB"
DISPLAY = "Rec.1886 Rec.709 - Display"
VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"

STYLES = ("ASC",)

#: Grades a colourist reaches for: every channel different from every other,
#: a lift and a crush, a power either side of one, and saturation both ways.
GRADES = (
    IDENTITY_GRADE,
    Grade(slope=(1.2, 1.0, 0.9), offset=(0.0, 0.0, 0.0), power=(1.0, 1.0, 1.0)),
    Grade(
        slope=(1.1, 0.95, 0.8),
        offset=(0.02, 0.0, -0.03),
        power=(1.2, 1.0, 0.85),
        saturation=1.25,
    ),
    Grade(saturation=0.0),
)


@pytest.fixture(scope="module", params=STYLES)
def graded(request, config, config_uri):
    return resolve_display_view(
        config, DISPLAY, VIEW, src=SOURCE, uri=config_uri, grade=request.param
    )


def test_the_processor_carries_no_grade(graded):
    """The render is resolved alone; the grade leads it in the artifact."""
    assert "CDL" not in op_names(graded.processor)


def test_the_graph_declares_the_grade_at_its_identity(graded):
    model = compile_processor(graded)
    knobs = parameters(model)
    assert list(knobs)[:4] == [CDL_SLOPE, CDL_OFFSET, CDL_POWER, CDL_SATURATION]
    np.testing.assert_array_equal(knobs[CDL_SLOPE], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(knobs[CDL_OFFSET], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(knobs[CDL_POWER], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(knobs[CDL_SATURATION], [1.0])


def test_the_artifact_records_its_grade_style(graded):
    metadata = {e.key: e.value for e in compile_processor(graded).metadata_props}
    assert metadata[GRADE_KEY] == graded.grade


def test_an_ungraded_render_records_no_grade(config, config_uri):
    plain = resolve_display_view(config, DISPLAY, VIEW, src=SOURCE, uri=config_uri)
    metadata = {e.key: e.value for e in compile_processor(plain).metadata_props}
    assert GRADE_KEY not in metadata
    assert plain.grade is None


def test_the_public_entry_point_takes_the_grade(config_uri):
    model = compile_display_view(DISPLAY, VIEW, src=SOURCE, grade="ASC")
    assert CDL_SLOPE in parameters(model)


@pytest.mark.parametrize("style", ["FILM", "NO_CLAMP"])
def test_a_style_with_no_path_is_refused(config, config_uri, style):
    """``NO_CLAMP`` included (§road:unclamped-grade): unclamped, a grade hands
    the render values past its finite range, where the graph and OCIO part
    company."""
    with pytest.raises(AddressError, match="grade"):
        resolve_display_view(
            config, DISPLAY, VIEW, src=SOURCE, uri=config_uri, grade=style
        )


def test_the_reference_leads_with_the_grade(graded):
    """At a grade off its identity the reference runs a CDL first."""
    reference = graded.graded(GRADES[2])
    assert op_names(reference)[0] == "CDL"


@pytest.mark.parametrize("grade", GRADES, ids=range(len(GRADES)))
def test_the_graph_tracks_ocio_across_grades(graded, grade):
    result = verify(graded, grade=grade)
    assert result.ok, str(result)


@pytest.mark.parametrize("grade", GRADES[1:], ids=range(1, len(GRADES)))
def test_a_grade_moves_the_picture(graded, grade):
    """The knob is wired: the reference at a grade differs from the identity's,
    so a graph ignoring its inputs could not track it."""
    from ocio_codegen.oracle import cpu_reference, lattice

    samples = lattice(graded.processor)
    assert not np.allclose(
        cpu_reference(graded.graded(grade), samples),
        cpu_reference(graded.graded(IDENTITY_GRADE), samples),
    )


class TestKernelSource:
    def test_graded_entry_points_replace_the_plain_ones(self, graded):
        """A consumer launching the plain entry on a graded kernel fails to
        find it, rather than running it with an argument missing."""
        source = cuda.kernel_source(graded)
        assert "apply_cdl_f32" in source
        assert "apply_cdl_f16" in source
        assert "void apply_f32(" not in source

    def test_the_grade_style_is_in_the_header(self, graded):
        source = cuda.kernel_source(graded)
        assert f"// {GRADE_KEY}: {graded.grade}" in source

    def test_compiles_without_a_device(self, graded):
        cuda.compile_ptx(cuda.kernel_source(graded), "compute_75")


@pytest.fixture(scope="module")
def cuda_runtime():
    pytest.importorskip("cuda.bindings")
    try:
        cuda.device_arch()
    except Exception as exc:  # any miss means "no GPU here", not a failure
        pytest.skip(f"no usable CUDA runtime: {exc}")


@pytest.mark.usefixtures("cuda_runtime")
class TestKernel:
    @pytest.mark.parametrize("grade", GRADES, ids=range(len(GRADES)))
    def test_the_kernel_tracks_ocio_across_grades(self, graded, grade):
        result = cuda.verify(graded, grade=grade)
        assert result.ok, str(result)

    def test_the_half_entry_tracks_the_float_one(self, graded):
        from ocio_codegen.oracle import lattice

        source = cuda.kernel_source(graded)
        samples = lattice(graded.processor)
        grade = GRADES[2]
        f32 = cuda.run_kernel(source, samples, grade=grade)
        f16 = cuda.run_kernel(source, samples, precision="f16", grade=grade)
        assert np.isfinite(f16).all()
        assert np.abs(f16 - f32).max() < 1e-2

    def test_a_graded_kernel_needs_its_grade(self, graded):
        from ocio_codegen.oracle import lattice

        with pytest.raises(ValueError, match="grade"):
            cuda.run_kernel(cuda.kernel_source(graded), lattice())


def test_grade_values_match_ocio_setters():
    """The dataclass hands OCIO exactly the ten numbers it holds."""
    grade = GRADES[2]
    transform = grade.transform("ASC")
    assert transform.getStyle() == OCIO.CDL_ASC
    np.testing.assert_allclose(transform.getSlope(), grade.slope)
    np.testing.assert_allclose(transform.getOffset(), grade.offset)
    np.testing.assert_allclose(transform.getPower(), grade.power)
    assert transform.getSat() == pytest.approx(grade.saturation)
