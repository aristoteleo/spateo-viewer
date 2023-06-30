from stviewer.static_app import state, static_server

if __name__ == "__main__":
    state.selected_dir = None
    static_server.start()
