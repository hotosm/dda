"""Static files + `/oam-api/*` OAM proxy + `/tiles/{z}/{x}/{y}?url=` rio-tiler XYZ for tools/index.html."""

import argparse
import http.server
import socketserver
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OAM_UPSTREAM = "https://api.openaerialmap.org"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/oam-api/"):
            self._proxy("/oam-api/", OAM_UPSTREAM)
            return
        if self.path.startswith("/tiles/"):
            self._serve_tile()
            return
        super().do_GET()

    def _serve_tile(self) -> None:
        """`/tiles/{z}/{x}/{y}?url=<cog-url>` -> 256px PNG via rio-tiler."""
        from rio_tiler.errors import TileOutsideBounds
        from rio_tiler.io import Reader

        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "tiles":
            self.send_error(400, "expected /tiles/{z}/{x}/{y}")
            return
        try:
            z, x, y = int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            self.send_error(400, "z/x/y must be integers")
            return
        qs = urllib.parse.parse_qs(parsed.query)
        cog_path = self._resolve_local_cog(qs.get("url", [""])[0])
        if cog_path is None:
            self.send_error(400, "url must be same-origin and resolve to a file under --root")
            return
        try:
            with Reader(input=str(cog_path)) as reader:  # ty: ignore[missing-argument]
                img = reader.tile(x, y, z)
        except TileOutsideBounds:
            self.send_error(404, f"tile {z}/{x}/{y} outside {cog_path.name}")
            return
        png_bytes = img.render(img_format="PNG")
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(png_bytes)))
        self.end_headers()
        self.wfile.write(png_bytes)

    def _resolve_local_cog(self, url: str) -> Path | None:
        """Return the doc-root-relative file for a same-origin URL, or None."""
        if not url:
            return None
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            return None
        rel = parsed.path.lstrip("/")
        candidate = (Path(self.directory) / rel).resolve()
        root = Path(self.directory).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def _proxy(self, prefix: str, upstream: str) -> None:
        tail = self.path[len(prefix) :]
        url = f"{upstream}/{tail}"
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                status = response.status
        except urllib.error.HTTPError as exc:
            body = exc.read() or b""
            content_type = exc.headers.get("Content-Type", "text/plain")
            status = exc.code
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    handler = lambda *a, **kw: Handler(*a, directory=str(args.root), **kw)  # noqa: E731
    with ReusableTCPServer(("127.0.0.1", args.port), handler) as server:
        print(f"serving {args.root} on http://127.0.0.1:{args.port}")
        print(f"open   http://127.0.0.1:{args.port}/tools/")
        server.serve_forever()


if __name__ == "__main__":
    main()
