import getopt
import sys

from stviewer.explorer_app import state, static_server

if __name__ == "__main__":
    # upload anndata
    state.selected_dir = None

    static_server.start()
