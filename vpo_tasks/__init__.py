"""Task plugins. Importing this package registers every task into
:data:`vpo.task.REGISTRY` via import side-effects.

To add a task: drop a ``vpo_tasks/<name>.py`` that constructs a
:class:`vpo.task.Task` and calls ``register(...)``, then add one import line
below. Nothing in ``vpo/`` needs to change.
"""

from vpo.task import REGISTRY, resolve  # re-export for convenience

from . import maze  # noqa: F401
from . import musique  # noqa: F401
from . import eureqa  # noqa: F401
from . import tool  # noqa: F401
from . import livecodebench  # noqa: F401

__all__ = ["REGISTRY", "resolve"]
