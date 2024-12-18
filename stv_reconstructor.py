import getopt
import sys

from stviewer.reconstructor_app import interactive_server, state

if __name__ == "__main__":
    # upload anndata
    state.upload_anndata = None
    opts, args = getopt.getopt(sys.argv[1:], "p", ["port="])
    port = "8888" if len(opts) == 0 else opts[0][1]
    interactive_server.start(port=port)
