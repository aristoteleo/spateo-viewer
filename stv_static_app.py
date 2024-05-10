import getopt
import sys

from stviewer.static_app import state, static_server

if __name__ == "__main__":
    # upload anndata
    state.selected_dir = None

    opts, args = getopt.getopt(sys.argv[1:], "p", ["port="])
    port = "1234" if len(opts) == 0 else opts[0][1]
    static_server.start(port=port, timeout=0)
