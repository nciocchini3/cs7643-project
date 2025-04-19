import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

# nv = good
# blk = good
# df = good
# vasc = good
# mel = bad
# bcc = bad
# akiec = bad

def main():
    metadata = pd.read_csv('archive/HAM10000_metadata.csv')

    nonDupCount = {}

    for index, row in metadata.iterrows():
        type = row['dx']

        if type not in nonDupCount:
            nonDupCount[type] = 0
            
        nonDupCount[type] = nonDupCount[type] + 1

    categories = ['nv', 'blk', 'df', 'vasc', 'mel', 'bcc', 'akiec']
    values = []
    for category in categories:
        print(category)
        print(categories)
        values.append(nonDupCount[category])
        
    plt.bar(categories, values)
    plt.show()


    #duplicates = metadata['lesion_id'].duplicated(keep=False)

    #dups = metadata[duplicates]
    #uniqueIds = dups['lesion_id'].unique()

    #for unqId in uniqueIds:
    #    rows = dups[dups['lesion_id'] == unqId]   
    #    imagesIds = rows['image_id']
    #    numImages = len(imagesIds)
    #    leasionId = rows['lesion_id']
    #    leasionType = rows['dx']
    #    leasionHow = rows['dx_type']
                
        
                
        #fig, axes = plt.subplots(numImages, figsize=(10, 8))
        
        #for index, image in enumerate(imagesIds):
        #    imageName = image + ".jpg"
        #    
        #    try:
        #        fullImage = mpimg.imread(os.path.join("archive/HAM10000_images_part_1/", imageName))
        #    except FileNotFoundError:
        #        try:
        #            fullImage = mpimg.imread(os.path.join("archive/HAM10000_images_part_2/", imageName))
        #        except FileExistsError:
        #            print("Could not find image for: ", imageName)
        #            continue
                
        #    ax = axes[index]
        #    ax.imshow(fullImage)
        #    ax.set_title(image) # Set title
        #    ax.axis('off') # Hide axes

        #plt.tight_layout()
        #filename = leasionId.values[0] + "_" + leasionType.values[0]
        #fig.savefig("images/" + filename + ".png")

main()