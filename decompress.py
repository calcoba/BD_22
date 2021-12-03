import bz2
import glob
import zipfile

path = './data/'

#compressed_file_path = glob.glob(path+'*.csv.bz2')

def decompress_bz2(compressed_file_path):
   print("Compressed files: ",compressed_file_path)
   for filename in compressed_file_path:
      zipfile = bz2.BZ2File(filename) # open the file
      data = zipfile.read() # get the decompressed data
      newfilepath = filename[:-4] # assuming the filepath ends with.bz2
      print("Created file: ",newfilepath)
      open(newfilepath, 'wb').write(data) # write a uncompressed file

def decompress_zip(compressed_folder_path):
   print("Compressed folder: ",compressed_folder_path)
   for filename in compressed_folder_path:
       print(filename)
       with zipfile.ZipFile(filename, 'r') as zipObj:
           zipObj.extractall('./data')