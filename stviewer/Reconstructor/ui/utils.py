from trame.widgets import html, vuetify

# -----------------------------------------------------------------------------
# vuetify components
# -----------------------------------------------------------------------------


def button(click, icon, tooltip):
    """Create a vuetify button."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with vuetify.VBtn(icon=True, v_bind="attrs", v_on="on", click=click):
                vuetify.VIcon(icon)
        html.Span(tooltip)


def checkbox(model, icons, tooltip, **kwargs):
    """Create a vuetify checkbox."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with html.Div(v_on="on", v_bind="attrs"):
                vuetify.VCheckbox(
                    v_model=model,
                    on_icon=icons[0],
                    off_icon=icons[1],
                    dense=True,
                    hide_details=True,
                    classes="my-0 py-0 ml-1",
                    **kwargs
                )
        html.Span(tooltip)


def switch(model, tooltip, **kwargs):
    """Create a vuetify switch."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            vuetify.VSwitch(v_model=model, hide_details=True, dense=True, **kwargs)
        html.Span(tooltip)
