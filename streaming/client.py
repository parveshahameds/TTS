import socket
import time
import threading
import queue
from streaming.protocol import TYPE_START, TYPE_DATA, TYPE_END, TYPE_PING, create_packet, COMPRESSION_NONE, COMPRESSION_ADPCM
from streaming.encoder import ImaAdpcmEncoder

class StreamingClient:
    def __init__(self, host="127.0.0.1", port=5055, compression=COMPRESSION_NONE):
        self.host = host
        self.port = port
        self.compression = compression
        
        self.sock = None
        self.send_queue = queue.Queue()
        self.is_connected = False
        self.is_running = False
        self.thread = None
        
        self.encoder = ImaAdpcmEncoder()
        self.seq_num = 0
        self.is_streaming = False
        
    def connect(self):
        """Attempts to connect to the ASR server and starts the sender thread."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Disable Nagle's algorithm for minimal latency
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            self.is_running = True
            
            # Start background sending thread
            self.thread = threading.Thread(target=self._send_loop, daemon=True)
            self.thread.start()
            
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            print(f"Connected to ASR Server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection to ASR server failed: {e}")
            self.is_connected = False
            return False
            
    def disconnect(self):
        """Disconnects and stops helper threads."""
        self.is_running = False
        self.is_streaming = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.is_connected = False
        print("Disconnected from ASR Server.")
        
    def start_stream(self, pre_roll_pcm, trigger_timestamp):
        """Signals start of audio stream and transmits the pre-roll data."""
        if not self.is_connected:
            return False
            
        print("KWS Triggered! Starting stream transmission...")
        self.is_streaming = True
        self.seq_num = 0
        self.encoder.reset()
        
        # 1. Send START packet
        start_packet = create_packet(TYPE_START, self.compression, b"", self.seq_num, trigger_timestamp)
        self.send_queue.put(start_packet)
        self.seq_num += 1
        
        # 2. Compress and send pre-roll audio
        if len(pre_roll_pcm) > 0:
            self.send_audio(pre_roll_pcm, timestamp=trigger_timestamp - (len(pre_roll_pcm) / 16000.0))
            
        return True
        
    def send_audio(self, pcm_chunk, timestamp=None):
        """Compresses and sends a block of audio."""
        if not self.is_connected or not self.is_streaming:
            return
            
        if self.compression == COMPRESSION_ADPCM:
            payload = self.encoder.encode(pcm_chunk)
        else:
            # PCM - 16-bit raw
            payload = pcm_chunk.tobytes()
            
        data_packet = create_packet(TYPE_DATA, self.compression, payload, self.seq_num, timestamp)
        self.send_queue.put(data_packet)
        self.seq_num += 1
        
    def stop_stream(self, timestamp=None):
        """Signals end of audio stream."""
        if not self.is_connected or not self.is_streaming:
            return
            
        print("Stopping stream transmission.")
        self.is_streaming = False
        
        end_packet = create_packet(TYPE_END, self.compression, b"", self.seq_num, timestamp)
        self.send_queue.put(end_packet)
        self.seq_num += 1
        
    def _send_loop(self):
        """Background thread to transmit packets with low latency."""
        while self.is_running:
            try:
                packet = self.send_queue.get(timeout=0.1)
                if self.sock:
                    self.sock.sendall(packet)
                self.send_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Network send error: {e}")
                self.is_connected = False
                break
                
    def _heartbeat_loop(self):
        """Sends periodic ping packets while idle to maintain TCP connection."""
        while self.is_running:
            time.sleep(5.0)
            if self.is_connected and not self.is_streaming:
                ping_packet = create_packet(TYPE_PING, self.compression, b"PING", 0)
                self.send_queue.put(ping_packet)
