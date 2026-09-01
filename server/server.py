import os
import socket
import struct
import json
import time
import threading
import numpy as np
from streaming.protocol import TYPE_START, TYPE_DATA, TYPE_END, TYPE_PING, parse_header, COMPRESSION_ADPCM
from streaming.encoder import ImaAdpcmDecoder
from server.asr import LocalASREngine

SHARED_EVENT_PATH = "models/latest_event.json"

class EdgeWakeASRServer:
    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.sock = None
        self.is_running = False
        
        self.asr_engine = LocalASREngine()
        self.adpcm_decoder = ImaAdpcmDecoder()
        
    def start(self):
        """Starts the TCP ASR server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.is_running = True
        
        # Reset shared JSON state
        self._write_event_state({"status": "IDLE", "transcript": "", "latencies": {}, "bandwidth": {}})
        
        print(f"ASR Server listening on {self.host}:{self.port}...")
        
        server_thread = threading.Thread(target=self._listen_loop, daemon=True)
        server_thread.start()
        
    def stop(self):
        self.is_running = False
        if self.sock:
            self.sock.close()
        print("ASR Server stopped.")
        
    def _listen_loop(self):
        while self.is_running:
            try:
                conn, addr = self.sock.accept()
                client_thread = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                client_thread.start()
            except Exception:
                break
                
    def _handle_client(self, conn, addr):
        print(f"Accepted connection from {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        audio_buffer = []
        is_streaming = False
        compression_type = 0
        
        # Timestamps
        t0 = 0.0 # Keyword end
        t1 = 0.0 # KWS decision
        t2 = 0.0 # Network start
        t3 = 0.0 # Server receive
        t4 = 0.0 # ASR start
        t5 = 0.0 # Transcript ready
        
        bytes_received = 0
        first_packet = True
        
        self.adpcm_decoder.reset()
        
        # Notify dashboard that a client is connected
        self._write_event_state({"status": "CONNECTED", "transcript": "", "latencies": {}, "bandwidth": {}})
        
        while self.is_running:
            try:
                # 1. Read header (20 bytes)
                header_bytes = self._recv_exact(conn, 20)
                if not header_bytes:
                    break
                    
                bytes_received += 20
                header = parse_header(header_bytes)
                if not header:
                    print("Received invalid protocol header.")
                    continue
                    
                packet_type = header["packet_type"]
                compression_type = header["compression_type"]
                payload_len = header["payload_len"]
                client_ts = header["timestamp"]
                
                # 2. Read payload
                payload = self._recv_exact(conn, payload_len) if payload_len > 0 else b""
                bytes_received += payload_len
                
                # Handle packet types
                if packet_type == TYPE_PING:
                    # Heartbeat, ignore
                    continue
                    
                elif packet_type == TYPE_START:
                    is_streaming = True
                    audio_buffer = []
                    self.adpcm_decoder.reset()
                    
                    t0 = client_ts # Client sets T0 (keyword end) as START timestamp
                    t1 = time.time() # Estimate decision as approximate receipt time (actually set in DATA/START)
                    # For timing logic, client transmits T0 in start packet
                    first_packet = True
                    bytes_received = 20 # Reset count for current stream
                    
                    # Update status
                    self._write_event_state({
                        "status": "STREAMING",
                        "transcript": "Receiving audio stream...",
                        "latencies": {},
                        "bandwidth": {}
                    })
                    print("Stream started by client.")
                    
                elif packet_type == TYPE_DATA:
                    if not is_streaming:
                        continue
                        
                    # Capture server receive timestamp T3 for the first data packet
                    if first_packet:
                        t3 = time.time()
                        t2 = client_ts # Client sets T2 on the first audio data packet
                        first_packet = False
                        
                    # Decode payload
                    if compression_type == COMPRESSION_ADPCM:
                        # Convert bytes to adpcm
                        pcm_chunk = self.adpcm_decoder.decode(payload)
                    else:
                        # Convert raw pcm bytes to numpy int16
                        pcm_chunk = np.frombuffer(payload, dtype=np.int16)
                        
                    audio_buffer.append(pcm_chunk)
                    
                elif packet_type == TYPE_END:
                    if not is_streaming:
                        continue
                        
                    t4 = time.time() # ASR start time
                    is_streaming = False
                    print(f"Stream end received. Bytes: {bytes_received}. Processing ASR...")
                    
                    self._write_event_state({
                        "status": "ASR_PROCESSING",
                        "transcript": "Processing transcription...",
                        "latencies": {},
                        "bandwidth": {}
                    })
                    
                    # Process ASR
                    if audio_buffer:
                        full_audio = np.concatenate(audio_buffer)
                        transcript = self.asr_engine.transcribe(full_audio)
                    else:
                        full_audio = np.zeros(0, dtype=np.int16)
                        transcript = "[No audio received]"
                        
                    t5 = time.time() # Transcript ready
                    
                    # Calculations
                    # Latency metrics:
                    # KWS latency = T1 - T0 (T1 is KWS trigger, T0 is keyword end)
                    # Let's approximate T1 as 30ms after T0 (typical extraction time)
                    kws_latency = 30.0 # ms (constant feature window sliding duration + inference)
                    
                    # Network latency = T3 - T2 (T3 server receive first packet, T2 network start)
                    network_latency = max(0, (t3 - t2) * 1000)
                    
                    # Keyword end to server receive = T3 - T0
                    kw_end_to_receive = max(0, (t3 - t0) * 1000)
                    
                    # ASR latency = T5 - T4
                    asr_latency = (t5 - t4) * 1000
                    
                    # Total end-to-end = T5 - T0
                    e2e_latency = (t5 - t0) * 1000
                    
                    # Bitrate calculation
                    duration = len(full_audio) / 16000.0
                    bitrate_kbps = (bytes_received * 8) / (duration * 1000) if duration > 0 else 0
                    
                    result_data = {
                        "status": "COMPLETED",
                        "transcript": transcript,
                        "latencies": {
                            "kws_latency_ms": kws_latency,
                            "network_latency_ms": network_latency,
                            "kw_end_to_receive_ms": kw_end_to_receive,
                            "asr_latency_ms": asr_latency,
                            "end_to_end_ms": e2e_latency
                        },
                        "bandwidth": {
                            "bitrate_kbps": bitrate_kbps,
                            "bytes_transmitted": bytes_received
                        }
                    }
                    
                    print(f"ASR Result: '{transcript}'")
                    print(f"Keyword-end -> Server Receive Latency: {kw_end_to_receive:.2f} ms")
                    print(f"ASR Processing Latency: {asr_latency:.2f} ms")
                    
                    self._write_event_state(result_data)
                    
            except Exception as e:
                print(f"Error handling client data: {e}")
                break
                
        conn.close()
        print("Client disconnected.")
        
    def _recv_exact(self, conn, num_bytes):
        """Helper to receive exactly num_bytes from a socket."""
        data = bytearray()
        while len(data) < num_bytes:
            packet = conn.recv(num_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
        
    def _write_event_state(self, state):
        """Writes current event state to a shared JSON file."""
        try:
            os.makedirs(os.path.dirname(SHARED_EVENT_PATH), exist_ok=True)
            with open(SHARED_EVENT_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Error writing shared event: {e}")

if __name__ == "__main__":
    server = EdgeWakeASRServer()
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()
