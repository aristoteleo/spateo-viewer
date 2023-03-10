from typing import Optional

from trame.app import get_server

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------


def get_trame_server(name: Optional[str] = None, **kwargs):
    """
    Return a server for serving trame applications. If a name is given and such server is not available yet, it will be
    created otherwise the previously created instance will be returned.

    Args:
        name: A server name identifier which can be useful when several servers are expected to be created.
        kwargs: Additional parameters that will be passed to ``trame.app.get_server`` function.

    Returns:
        Return a unique Server instance per given name.
    """

    server = get_server(name=name, **kwargs)
    return server
