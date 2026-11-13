"""Serve the built site (docs/site) locally, with the headers the playground needs, as Netlify does.

    python3 docs/serve.py [PORT]
"""

import functools
import http.server
import os
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # As docs/landing/_headers: cross-origin isolation, so the playground can share memory between threads
        if self.path.startswith("/playground"):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
    handler = functools.partial(Handler, directory=site)
    # Wasm needs the right MIME type to be compiled while it downloads
    handler.func.extensions_map[".wasm"] = "application/wasm"
    print(f"Serving {site} at http://localhost:{port}")
    http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
