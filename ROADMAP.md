# ocio-codegen — Roadmap

Derived from SPEC.md. Every item traces to a spec gap. Sections are
in build-dependency order. Completed work is removed; presence here
means the work is not done.

## Live parameters §road:live-parameters

The scalar and per-channel properties compile to graph inputs
(§spec:dynamic-properties), and a display render can lead with a live CDL
(§spec:graded-render). What is left is the grade style that render does not
yet take.

### Unclamped graded render §road:unclamped-grade

The `NO_CLAMP` style for a graded render (§spec:graded-render). Blocked on
the render graph's answer to inputs whose linear overflows float32: it
answers NaN where OCIO answers zero, on neutral log values above about 8.6
into the ACES 2.0 SDR render. The ungraded lattice never forms such a triple,
and an unclamped saturation does. Once the graph agrees there, the identity
case remains: OCIO removes an unclamped CDL at its identity, so the graph's
evaluated identity has to round the same way or skip the same way.

## Packaging §road:packaging

Publish the compiler as an installable package with the oracle harness
runnable by anyone (§spec:problem-statement). The audience is real-time
CUDA-resident video, which has had no answer here; making the artifact
easy to obtain and easy to check is most of what makes it useful to
someone who is not us.

### Distribution §road:distribution

Package for PyPI with `opencolorio` and `onnx` as the only runtime
dependencies, both permissively licensed, and pin the OCIO version range
the op coverage was measured against (§spec:op-coverage).

### Coverage report §road:coverage-report

Emit the op census and the verify/refuse partition as a generated artifact
per OCIO release, so a consumer reads what is supported from measurement
rather than from prose (§spec:op-coverage, §spec:verification). This is
also how an OCIO upgrade surfaces a new op: the partition changes.

## CUDA executor coverage §road:cuda-executor

Hold every emitted graph on ONNX Runtime's CUDA executor with CPU fallback
off (§spec:verification). All but the ACES 2.0 views run there now.

### Atan on the CUDA executor §road:cuda-atan

The 24 ACES 2.0 view graphs each carry one `Atan`, the hue angle, and the
CUDA executor has no kernel for it, so those sessions are refused. Emit the
angle from ops the executor places, held to the oracle at the graph's own
tolerance, or take an ONNX Runtime release that adds the kernel; either way
the pinned refusal count in `tests/test_cuda_provider.py` falls to zero.

## Apple GPU execution §road:apple-gpu

Hold the emitted graph on an Apple GPU the way §spec:verification holds it
on the CUDA executor: every node on the device, the same lattice, the same
reference. ONNX Runtime's CoreML executor is the path a consumer on that
hardware takes. With CPU fallback off it accepts none of the pinned
config's 159 graphs: 140 fail CoreML's ML Program parse at the 1×1
convolution, 8 at a `Clip`, and 11 carry an op CoreML does not place.

### CoreML acceptance §road:coreml-acceptance

Emit the convolution and the clamp in forms CoreML's ML Program accepts, or
refuse by name where none exists, and pin the accepted count as a census
(§spec:op-coverage, §spec:verification). The test runs on a self-hosted
Apple Silicon runner, because hosted macOS runners expose no GPU.
