from matplotlib import pyplot as plt
from skimage import data, feature, color, exposure, filters
import skimage.morphology as mp
import math
files = ['samolot01.jpg','samolot07.jpg','samolot08.jpg','samolot09.jpg','samolot10.jpg','samolot16.jpg']
plt.figure(figsize=(30, 20))
for i, file in enumerate(files):
    img = data.imread(file, True)
    img = color.rgb2gray(img)
    img = feature.canny(img, sigma=6)
    img = filters.sobel(img)
    img = exposure.adjust_sigmoid(img, 0.1, 1000)
    img = mp.dilation(img, mp.disk(1))
    plt.subplot(math.ceil(len(files)/3), 3, i + 1)
    plt.axis('off')
    plt.imshow(img, cmap="gray")

plt.tight_layout(pad=0)
plt.savefig('Contours_generated.jpg')