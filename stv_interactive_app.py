from stviewer.interactive_app import interactive_server, state

if __name__ == "__main__":
    # upload anndata
    state.upload_anndata = None
    # host="0.0.0.0", port="8080"
    interactive_server.start(port="8080")
