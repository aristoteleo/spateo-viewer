from stviewer.interactive_app import interactive_server, state

if __name__ == "__main__":
    state.upload_anndata = None
    interactive_server.start()
