import os
import qualityface

directory = 'test'
images = os.listdir(directory)

for img in images:
    path = os.path.join(directory, img)
    score = qualityface.estimate(path)
    print(f"Test: {img}, score: {score}")