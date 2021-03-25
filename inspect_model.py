# import necessary package
from keras.applications import VGG16
import argparse

# construct argument parse and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1,
help="whether or not to include top of CNN")
args = vars(ap.parse_args())


# load the VGG16 network
print("[INFO] loading network ...")
model = VGG16(weights="imagenet", 
include_top=args["include_top"] > 0)
print("[INFO] showing layers ...")

# loop over the layers in the network and
# display them to the console 
for (i, layer) in enumerate(model.layers):
    print(f"[INFO] {i}\t{layer.__class__.__name__}")