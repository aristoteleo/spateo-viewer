from trame.widgets import html, trame, vuetify

from stviewer.static_viewer.pv_pipeline.init_parameters import (
    init_mesh_parameters,
    init_morphogenesis_parameters,
    init_pc_parameters,
)


def pipeline_content(server, plotter):
    # server
    state, ctrl = server.state, server.controller

    @ctrl.set("actives_change")
    def actives_change(ids):
        _id = ids[0]
        active_actor_id = state.actor_ids[int(_id) - 1]
        state.active_ui = active_actor_id
        state.active_model_type = str(state.active_ui).split("_")[0]
        state.active_id = int(_id)

        if state.active_ui.startswith("PC"):
            state.update(init_pc_parameters)
            state.update(init_morphogenesis_parameters)
        elif state.active_ui.startswith("Mesh"):
            state.update(init_mesh_parameters)
        ctrl.view_update()

    @ctrl.set("visibility_change")
    def visibility_change(event):
        _id = event["id"]
        _visibility = event["visible"]
        active_actor = [value for value in plotter.actors.values()][int(_id) - 1]
        active_actor.SetVisibility(_visibility)
        if _visibility is True:
            state.vis_ids.append(int(_id) - 1)
        else:
            state.vis_ids.remove(int(_id) - 1)
        state.vis_ids = list(set(state.vis_ids))
        ctrl.view_update()

    # main content
    trame.GitTree(
        sources=("pipeline",),
        actives_change=(ctrl.actives_change, "[$event]"),
        visibility_change=(ctrl.visibility_change, "[$event]"),
    )


def pipeline_panel(server, plotter):
    # Logo and title
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        # Logo and title
        vuetify.VIcon("mdi-source-branch", style="transform: scale(1, -1);")
        vuetify.VCardTitle(
            " Pipeline",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

    # Main content
    pipeline_content(server=server, plotter=plotter)
