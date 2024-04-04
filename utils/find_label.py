import sys
import struct

def idx_header(file):
    magic = struct.unpack('>BBBB', file.read(4))
    assert magic[0] == 0 and magic[1] == 0 and magic[2] == 0x08 and magic[3] == 1  # idx unsigned byte
    (size) =  struct.unpack('>I', file.read(4))
    return size
    

def main(argc, argv):
    pos = int(argv[3])
    label = bytes(argv[2], 'ascii')
    idx = argv[1]
    with open(idx, 'rb') as f:
        size = idx_header(f)
        header_sz = f.tell()
        f.seek(pos + header_sz)
        while(True):
            b = f.read(1)
            if not b:
                print("Not found", pos, label)
                break
            if b == label:
                print(f.tell() - header_sz - 1)
                break
            pos += 1
    return 0

if __name__ == '__main__':
    exit(main(len(sys.argv), sys.argv))
