from api.index import app
from wsgiref.simple_server import make_server

if __name__ == "__main__":
    port = 3000
    print(f"Starting EdgeWake Web Server on http://localhost:{port} ...")
    with make_server("0.0.0.0", port, app) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
