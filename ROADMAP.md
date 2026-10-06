# ocio-codegen — Roadmap

Derived from SPEC.md. Every item traces to a spec gap. Sections are
in build-dependency order. Completed work is removed; presence here
means the work is not done.

## Live parameters §road:live-parameters

The scalar and per-channel properties compile to graph inputs
(§spec:dynamic-properties), so a consumer varies a grade per frame without
recompiling. What is left is the shape those inputs cannot take.

### Curve-shaped grading properties §road:grading-curves

The four `GRADING_*` families, whose parameters are control points rather
than values (§spec:dynamic-properties). Deferred behind a named trigger:
the first consumer that needs curve-based grading rather than CDL. Until
then the compiler refuses them by the same mechanism as any unimplemented
op (§spec:op-coverage).

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
