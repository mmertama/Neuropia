import sys
import struct

if __name__ == '__main__':
    bin_name = sys.argv[1]
    p_at = int(sys.argv[2])
    p_len = int(sys.argv[3])
    data = None
    with open(bin_name, 'rb') as file:
        file.seek(p_at)
        data = file.read(p_len);
        if not data:
            print("Nothing")
            exit(0)

        format_string = f"={p_len}s"
        value = struct.unpack(format_string, data)[0]
        print([hex(v) for v in value])
        print([chr(v) for v in value])

     



