import sys
import zipfile
import pathlib
import struct
from PIL import Image
import io

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
    assert ascii > ord(' ')
    assert ascii < 255
    byte = ascii.to_bytes(1, byteorder='big')
    return byte
    

def progress_bar(progress, all, bar_length = 60):
    percent = float(progress) / all
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    print("\r[{0}] {1}%".format(hashes + spaces, int(round(percent * 100))), end='')
    
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
        elen = len(entries)
        hsf_images, hsf_labels = create_idx("hsf-" + name_prefix, target_folder, elen, w, h)
        train_images, train_labels = create_idx("train-" + name_prefix, target_folder, elen, w, h)
        progress = 0
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
                progress_bar(progress, elen)
                progress += 1  
                labels.write(label_byte(ascii))
                bytes = infile.read()
                images.write(image_bytes(bytes, w, h))
    print("\nhsf: ", hsf_entries, "train: ", train_entries, "classes: ", len(label_set), " ", [chr(x) for x in label_set]) # newline when progressbar is gone