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
