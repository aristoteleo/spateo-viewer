# -----------------------------------------------------------------------------
# GUI layout
# -----------------------------------------------------------------------------


def ui_layout(server, template_name: str = "main", drawer_width: int = 300, **kwargs):
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

    return SinglePageWithDrawerLayout(
        server, template_name=template_name, width=drawer_width, **kwargs
    )
