import random, os
from PIL import Image
import numpy as np

INPATH = r"/Users/lirongazit/Documents/dataset"
OUTPATH = r"/Users/lirongazit/Documents/outcropDataset"

dx = dy = 128
tilesPerImage = 100

files = os.listdir(INPATH)
numOfImages = len(files)

for file in files:
   with Image.open(os.path.join(INPATH, file)) as im:
     for i in range(1, tilesPerImage+1):
       try:
         newname = file.replace('.', '_{:03d}.'.format(i))
         w, h = im.size
         x = random.randint(0, w-dx-1)
         y = random.randint(0, h-dy-1)
         print("      Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
         im.crop((x,y, x+dx, y+dy))\
           .save(os.path.join(OUTPATH, newname))
       except:
         pass

print("Done {}".format(numOfImages))

files = os.listdir(OUTPATH)

for file in files:

  im = Image.open(r"/Users/lirongazit/Documents/outputcropDataset/"+file)
  im = (np.array(im))

  r = im[:,:,0].flatten()
  g = im[:,:,1].flatten()
  b = im[:,:,2].flatten()
  label = [1]
  out = np.array(list(r) + list(g) + list(b),np.uint8)
  out = np.array(list(label),np.uint8)
  out.tofile(r"/Users/lirongazit/Documents/outimage.gz")
  out.tofile(r"/Users/lirongazit/Documents/oulabel.gz")


#out.tofile(r"/Users/lirongazit/Documents/out.bin")

