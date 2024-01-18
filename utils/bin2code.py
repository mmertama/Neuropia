import sys
import struct
import os

if __name__ == '__main__':
    bin_name = sys.argv[1]
    out_name = sys.argv[2]
    data_name = sys.argv[3]

    file_stats = os.stat(bin_name)
    assert file_stats.st_size % 4, "File size should be module of 4"
    item_len = 0
    with open(out_name, 'w') as out:
        with open(bin_name, 'rb') as file:
            print('// this file is generated\n', file=out)
            print('constexpr uint8_t ', data_name, '[] = {\n', file=out)
            while True:
                data = file.read(1);

                if not data:
                    break

                value = struct.unpack('=B', data)[0]
                print(hex(value), file=out, end=',')
                item_len += 1
                if not item_len % 20:
                    print("", file=out) # new line
            print('\n};', file=out)



