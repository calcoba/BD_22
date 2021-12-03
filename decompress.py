import bz2
import zipfile

path = './data/'


# compressed_file_path = glob.glob(path+'*.csv.bz2')


def decompress_bz2(compressed_file_path):
    print("Compressed files: ", compressed_file_path)
    for filename in compressed_file_path:
        zip_file = bz2.BZ2File(filename)  # open the file
        data = zip_file.read()  # get the decompressed data
        new_filepath = filename[:-4]  # assuming the filepath ends with.bz2
        print("Created file: ", new_filepath)
        open(new_filepath, 'wb').write(data)  # write a uncompressed file


def decompress_zip(compressed_folder_path):
    print("Compressed folder: ", compressed_folder_path)
    for filename in compressed_folder_path:
        print(filename)
        with zipfile.ZipFile(filename, 'r') as zipObj:
            zipObj.extractall('./data')
