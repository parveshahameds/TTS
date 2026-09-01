from api.index import handler, app

if __name__ == "__main__":
    from http.server import HTTPServer
    print("Starting EdgeWake Web Server on http://localhost:3000 ...")
    server = HTTPServer(("0.0.0.0", 3000), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()
