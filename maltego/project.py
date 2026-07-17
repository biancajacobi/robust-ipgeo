"""maltego-trx-Einstiegspunkt (lokale Transforms).

Registrierung in Maltego siehe maltego/README.md. Direktaufruf zum Testen:

  python project.py list                          # registrierte Transforms
  python project.py local robustgeolocate 9.9.9.9
  python project.py local persourceestimates 9.9.9.9
"""

import sys

import transforms  # noqa: F401  (Registry sammelt die Transform-Klassen ein)
from maltego_trx.handler import handle_run
from maltego_trx.registry import register_transform_classes
from maltego_trx.server import app, application  # noqa: F401  (fuer optionalen Server-Betrieb)

register_transform_classes(transforms)

handle_run(__name__, sys.argv, app)
