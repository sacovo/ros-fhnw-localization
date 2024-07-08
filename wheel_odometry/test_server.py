import socket
import itertools
import random
import struct
import sys
import time

# Define server details
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = int(sys.argv[1])       # Change this to a desired port number
INTERVAL = 1       # Interval between sending data (seconds)
NUM_INTS = 15      # Number of random integers to send

def main():
  # Create a TCP socket
  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server_socket.bind((HOST, PORT))
  server_socket.listen()

  print(f"Server listening on {HOST}:{PORT}")

  while True:
    # Wait for a client connection
    client_socket, address = server_socket.accept()
    print(f"Connected by {address}")

    try:
      while True:
        # Generate 15 random integers
        data = [random.randint(0, 100) for _ in range(NUM_INTS)]

        # Pack integers into a byte array
        data_bytes = b''.join(itertools.chain([x.to_bytes(4, 'big', signed=True) for x in data]))
        print(data_bytes)

        # Send data to the client
        client_socket.sendall(data_bytes)

        # Wait for the interval
        print(f"Sent {NUM_INTS} random integers")
        print(', '.join([str(x) for x in data]))
        client_socket.settimeout(INTERVAL)
        time.sleep(INTERVAL)
    except (socket.timeout, ConnectionResetError):
      print(f"Client {address} disconnected")
      client_socket.close()
    finally:
      client_socket.close()

if __name__ == "__main__":
  main()