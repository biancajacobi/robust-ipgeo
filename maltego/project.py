"""maltego-trx entry point (local transforms).

See maltego/README.md for registering the transforms in Maltego. Direct
invocation for testing:

  python project.py list                          # registered transforms
  python project.py local robustgeolocate 9.9.9.9
  python project.py local persourceestimates 9.9.9.9
"""

import sys

import transforms  # noqa: F401  (registry collects the transform classes)
from maltego_trx.handler import handle_run
from maltego_trx.registry import register_transform_classes
from maltego_trx.server import app, application  # noqa: F401  (for optional server mode)

register_transform_classes(transforms)

handle_run(__name__, sys.argv, app)
