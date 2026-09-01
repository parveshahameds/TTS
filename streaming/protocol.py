import struct
import time

# Packet types
TYPE_START = 0x01
TYPE_DATA  = 0x02
TYPE_END   = 0x03
TYPE_PING  = 0x04

# Compression options
COMPRESSION_NONE   = 0x00 # Raw PCM (16-bit)
COMPRESSION_ADPCM  = 0x01 # IMA ADPCM (4-bit)

# Struct format: 2s (Magic) + B (Type) + B (Compression) + I (Length) + d (Timestamp in seconds)
# Size = 2 + 1 + 1 + 4 + 8 = 16 bytes
HEADER_FORMAT = "!2sBBII"
MAGIC_BYTES = b"EW"

def create_packet(packet_type, compression_type, payload, seq_num=0, timestamp=None):
    """Creates a binary packet with a 16-byte header."""
    if timestamp is None:
        timestamp = time.time()
        
    # Convert timestamp to milliseconds since epoch as integer (to avoid float issues)
    ts_ms = int(timestamp * 1000)
    
    header = struct.pack(HEADER_FORMAT, MAGIC_BYTES, packet_type, compression_type, seq_num, len(payload))
    # We append the 8-byte timestamp in millisecond integer format
    ts_bytes = struct.pack("!Q", ts_ms)
    return header + ts_bytes + payload

def parse_header(header_data):
    """Parses a 20-byte header (12 bytes format + 8 bytes timestamp)."""
    if len(header_data) < 20:
        return None
        
    magic, packet_type, compression_type, seq_num, payload_len = struct.unpack(HEADER_FORMAT, header_data[:12])
    ts_ms = struct.unpack("!Q", header_data[12:20])[0]
    
    if magic != MAGIC_BYTES:
        return None
        
    return {
        "packet_type": packet_type,
        "compression_type": compression_type,
        "seq_num": seq_num,
        "payload_len": payload_len,
        "timestamp": ts_ms / 1000.0
    }
