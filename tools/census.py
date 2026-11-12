#!/usr/bin/env python3
"""Report a config's op coverage (§spec:op-coverage, §road:coverage-report).

The measurement lives in ``ocio_codegen.census``, beside the compiler whose
supported set it reads, and is also reachable as ``ocio_codegen census``. This
script stays because SPEC.md names it by path. Requires the package
installed. Usage::

    python tools/census.py [config-uri]
"""

import sys

from ocio_codegen.census import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
