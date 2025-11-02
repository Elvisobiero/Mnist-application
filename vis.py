import os
import random
import cv2 as cv
import matplotlib.pyplot as plt


dir = 'trainingSet'
categories = os.listdir(dir)
categories.sort()

vis_data = []

for category in categories:
    path = os.path.join(dir, category)
    for image in os.listdir(path)[:2]:
        img = cv.imread(os.path.join(path, image))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        vis_data.append([img, category])
        
        
random.shuffle(vis_data)


plt.figure(figsize=(12, 6))
for i, (img, category) in enumerate(vis_data):
    plt.subplot(4, 5, i+1)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")
    
plt.tight_layout()
plt.show()
