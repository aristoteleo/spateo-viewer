from typing import Optional

# -----------------------------------------------------------------------------
# GUI layout
# -----------------------------------------------------------------------------


def ui_layout(
    server, template_name: str = "main", drawer_width: Optional[int] = None, **kwargs
):
    """
    Define the user interface (UI) layout.
    Reference: https://trame.readthedocs.io/en/latest/trame.ui.vuetify.html#trame.ui.vuetify.SinglePageWithDrawerLayout

    Args:
        server: Server to bound the layout to.
        template_name: Name of the template.
        drawer_width: Drawer width in pixel.

    Returns:
        The SinglePageWithDrawerLayout layout object.
    """
    from trame.ui.vuetify import SinglePageWithDrawerLayout

    if drawer_width is None:
        screen_width, screen_height = (1920, 1080)
        drawer_width = int(screen_width * 0.15)

    return SinglePageWithDrawerLayout(
        server, template_name=template_name, width=drawer_width, **kwargs
    )
