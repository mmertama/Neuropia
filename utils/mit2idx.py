import sys
import zipfile
import pathlib
import struct
from PIL import Image
import io

VALUES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
              'W', 'X', 'Y', 'Z', 
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
              'w', 'x', 'y', 'z']

values = [VALUES.index(x) if x in VALUES else -1 for x in map(chr, range(256))]

def create_idx(prefix, folder, size: int, width: int, height: int):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    image_file = open(folder + '/' + prefix + "-images-idx3-ubyte.idx", 'wb')
    image_magic = struct.pack('>BBBB', 0, 0, 0x08, 3)  # idx unsigned byte
    image_file.write(image_magic)
    image_file.write(struct.pack('>III', size, width, height))
    label_file = open(folder + '/' + prefix + "-labels-idx1-ubyte.idx", 'wb')
    label_magic = struct.pack('>BBBB', 0, 0, 0x08, 1)  # idx unsigned byte
    label_file.write(label_magic)
    label_file.write(struct.pack('>I', size))
    return image_file, label_file
    
def image_bytes(image_data, width: int, height: int):
    image = Image.open(io.BytesIO(image_data))
    grayscale = image if image.mode == 'L' else image.convert('L')
    grayscale = grayscale.resize((width, height))
    bytes = grayscale.tobytes()
    assert len(bytes) == width * height, "except " + str(width * height) + " get " + str(len(bytes)) + " " + str(image.size)
    return bytes


def label_byte(ascii):
    global values   
    index = values[ascii]
    assert(index >= 0)
    byte = index.to_bytes(1, byteorder='big')
    return byte

def write(f, bytes):
    for b in bytes:
        f.write(b) 

def progress_bar(progress, all, bar_length = 60):
    fraction = float(progress) / all
    percent = int(round(fraction * 100))
    if progress_bar.pre_percent == percent:
        return
    progress_bar.pre_percent = percent
    hashes = '#' * int(round(fraction * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    print("\r[{0}] {1}%".format(hashes + spaces, percent), end='')
progress_bar.pre_percent = -1

if __name__ == "__main__":    
    zip_file = sys.argv[1]
    target_folder = sys.argv[2]
    name_prefix = sys.argv[3]
    w = int(sys.argv[4])
    h = int(sys.argv[5])
    label_set = set()
    train_entries = 0
    hsf_entries = 0
    with zipfile.ZipFile(zip_file, 'r') as zip:
        entries = zip.namelist()
        entries_len = len(entries)
        progress = 0
        hsf_images = []
        hsf_labels = []
        train_images = []
        train_labels = []
        for e in entries:
            path = e.split('/')
            if len(path[1]) == 0:
                continue
            if len(path[2]) == 0:
                continue
            if not e.lower().endswith('.png'):
                continue
            try:
                ascii = int(path[1], 16)
            except Exception as err:
                print(err, path)
                exit(1)
            label_set.add(ascii)    
            if path[2].startswith('train'):
               labels = train_labels
               images = train_images
               train_entries += 1
            else:
                labels = hsf_labels
                images = hsf_images
                hsf_entries += 1            
            with zip.open(e) as infile:
                progress_bar(progress, entries_len)
                progress += 1  
                labels.append(label_byte(ascii))
                bytes = infile.read()
                images.append(image_bytes(bytes, w, h))
    
    hsf_images_file, hsf_labels_file = create_idx("hsf-" + name_prefix, target_folder, hsf_entries, w, h)
    train_images_file, train_labels_file = create_idx("train-" + name_prefix, target_folder, train_entries, w, h)
    
    print("\nWrite files...")
    write(hsf_labels_file, hsf_labels)
    write(hsf_images_file, hsf_images)
    write(train_labels_file, train_labels)
    write(train_images_file, train_images)
    print("hsf: ", hsf_entries, "train: ", train_entries, "classes: ", len(label_set)) # newline when progressbar is gone
