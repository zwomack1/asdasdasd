import http.server
import socketserver
import threading

def run_http_server():
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start HTTP server in a separate thread
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()
    
    # Test making a request to the local server
    import requests
    try:
        response = requests.get("http://localhost:8000")
        print(f"HTTP request successful: {response.status_code}")
    except Exception as e:
        print(f"HTTP request failed: {str(e)}")
    
    # Keep the server running for a while
    server_thread.join(timeout=5)
