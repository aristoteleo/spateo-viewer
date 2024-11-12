import getopt
import sys

from stviewer.reconstructor_app import interactive_server, state

if __name__ == "__main__":
    # upload anndata
    state.upload_anndata = None
    interactive_server.start()
