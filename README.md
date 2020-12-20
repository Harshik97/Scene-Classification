# Scene Classification

This repositiory contains the code for developing a scene recognition pipeline. The pipeline is based on the Bag-of-words (BoW) model. Visual words are developed using the features extracted from training images. These words are then clustered to form a dictionary. Now, given a test image, one can extract its features to compute a visual word which can then be compared to all visual words present in the training dictionary. In this code, the comparison is done using histogram intersection similarity. The pair of test visual word and training visual word having the highest histogram instersection similarity is adjudged the matched pair.





The code implements the above described approach using Spatial Pyramid Matching. 
# References.
* ['Spatial Pyramid Matching for recognizing natural scene categories'](https://ieeexplore.ieee.org/document/1641019)
* ['Object categorization by learned universal visual dictionary.'](https://ieeexplore.ieee.org/document/1544935)
* Dataset: ['Large-scale scene recognition from abbey to zoo'](https://ieeexplore.ieee.org/document/5539970)

