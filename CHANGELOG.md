# 1.0.0 (2026-09-20)


### Bug Fixes

* **aces:** refuse the parameterisations that reached past the tables ([ed9c617](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/ed9c6178d1c427c1d9213184903ca057e4613f34))
* **addressing:** keep a graded op live through OCIO's optimizer ([80e398b](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/80e398b93ce97527f3452d242fbd0ff537c18491))
* **addressing:** report OCIO's lazy file errors as a refusal, not a crash ([f31749a](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/f31749aa9e21715efd39c9206d6a76c32ced4929))
* **builder:** declare a live parameter in constant time ([4ac8dc5](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/4ac8dc5bcd0b3a63d562e5d6c45755a1d0347d01))
* **lut1d:** bound the gather index so a NaN pixel cannot leave the table ([8d203e7](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/8d203e7f49d3b2886e62b19ee887a6653a65f589))
* **oracle:** check the samples array lines up with the reference ([abacfc8](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/abacfc845de29b6870b4fe820277c7ad94e555cf))
* **oracle:** copy the samples the reference runs over ([8d911a3](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/8d911a397d18a7fadafc6abae39e2a488f369459))
* **oracle:** make every lattice sample evidence, and require one finite ([d2ce0fb](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/d2ce0fb10c7fc63c6f7aa811a0773ce1bc715eaf))
* **oracle:** name the sample a class disagreement was found at ([a554a74](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/a554a744d34c808a39bebc845df1265c5b48c32f))
* **tests:** escape the path handed to `pytest.raises(match=)` ([1b698be](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/1b698be5b696f061cc23c35a8e71ef739ca7bf00))


### Features

* **addressing:** resolve a compile request against the config loaded ([55150c6](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/55150c66891aee5fa2ac5b59f08d3fcf7504d09a))
* **api:** compile a color space pair or a display view ([c91af49](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/c91af498a88e6a5bfeece58cd218b5b4eacfde6c))
* **builder:** accumulate ONNX nodes into a checked float32 graph ([632a569](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/632a569946b0853cbf90c778f30012e8a03d6952))
* **cli:** compile, verify, and measure from the command line ([a87da17](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/a87da178660a078b9df2bde29a71db6528c09644))
* **cli:** emit and verify the CUDA kernel beside the graph ([ab8e02e](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/ab8e02e3041304f9e3153027998faaeda84e1218))
* **compiler:** emit Matrix, and refuse the ops with no emitter ([e497be7](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/e497be745618e2ec84c92cb5de61fa589f13cc4d))
* **compiler:** refuse a processor's unimplemented ops before emitting ([3def931](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/3def931b46648d4f21bca6de5dd8857609ba40e8))
* **cuda:** transpile a processor's GPU shader to a fused kernel ([16c4433](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/16c443327dc285efa89150c9bbdb1364be9fe266))
* **emitters:** emit Exponent in both directions ([6780c2b](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/6780c2b7802f5b646ae3e653011d2700c45b01cf))
* **emitters:** emit ExponentWithLinear in both directions ([d6bf2c2](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/d6bf2c20a77cdca78a7acb3e644d0a9b4542ebdf))
* **emitters:** emit Log in both directions ([7049705](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/7049705aaf1d22f39c2f7097307cd6f196204038))
* **emitters:** emit LogCamera in both directions ([516de8d](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/516de8dea2126bd16cff00f969820da0aabf1135))
* **emitters:** emit Range as a clamp and an affine ([56eb5bf](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/56eb5bf3e9b4c2be7bb2990b4aee88381cc67b7d))
* **emitters:** emit REC2100_SURROUND, unblocking HLG display output ([497018b](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/497018bcad16e835541be38ce2c3518b899cb38e))
* **emitters:** emit scalar dynamic properties as graph inputs ([eca3a96](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/eca3a9600bbf08d517a26b3ee85b9715e87945c7))
* **emitters:** emit the ACES 2.0 output transform ([daac81e](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/daac81ef48212d98239f5a8daa28ce9446e48d5e))
* **lut1d:** emit a forward half-domain Lut1D by reconstructing the index ([925bfbf](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/925bfbfc2f83e9e8fefc2121b44d74e8c432f476))
* **lut1d:** emit a forward uniform Lut1D as a gather and a lerp ([7dda7bb](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/7dda7bb4eb3b8e5b082095edcb080af88cc49a48))
* **lut1d:** invert a monotonic table at compile time onto the half domain ([fd3b5ad](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/fd3b5ad79d0dbf63eea37c3d7ff36c68cf1ede98))
* **oracle:** verify an emitted graph against OCIO's CPU processor ([8c0f9d5](https://github.com/Fuse-Technical-Group/ocio-codegen/commit/8c0f9d59a6612658d4a5aaba764f89af0e1b97df))
