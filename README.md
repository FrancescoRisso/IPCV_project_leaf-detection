# IPCV_project_leaf-detection

Project for the course Image Processing and Computer Vision at Polytechnic of Turin, A.Y. 2023/24

Following there is the general report for the project.
If you need usage instructions, you can jump directly to [8. Usage Instructions](#8-usage-instructions)

## 1. Leaf Recognition

When immersed in nature, a casual glance around us reveals that we are surrounded by plants.
A more attentive eye might notice that they are not all the same; each plant has leaves of different shapes, sizes, and colors...
A curious eye might finally wonder to which species, among the nearly 400,000 existing on Earth, the plant they are looking at belongs.

Besides the shape of the tree, a big clue in recognizing a plant is the leaves: one can observe shapes, colors, sizes, veins...

This is what sparked our desire to create a project capable of distinguishing a plant given one of its leaves, considering for simplicity a small subset of plants.

## 2. Operational Choices

The state of the art for a task of this type is neural networks, which can self-learn a huge amount of features, then classify based on those.
However, our goal for this project was not to create an innovative system, better than what currently exists: we wanted, on the contrary, to "get our hands dirty" with what we learned during the course.

For this reason, we decided to avoid neural networks, on which we do not have particular knowledge.
Instead, we focused on analyzing images with traditional methods, to extract features such as measurements, shapes, convex hull...

These features are then discretized to be provided to a Bayesian classifier, which classifies the new images.

By adding human knowledge about the features to be learned, the Bayesian classifier has to learn much less compared to a neural network that works directly on images.
This allows for good results even starting from relatively small training sets.
This characteristic was very important to us, as we wanted to create our own dataset, which included the plants we see every day in our gardens: it was therefore not possible for us to destroy a plant by stealing hundreds of leaves, we had to limit ourselves to a few leaves per species.

## 3. Constraints

To be able to accurately calculate features related to dimensions, we needed the photos to have a fixed length reference.
This is to avoid errors related to the camera-leaf distance, which affects the number of pixels covered by the leaf.
For this reason, we decided that all photos should be taken of a single leaf, placed on an A4 sheet positioned vertically, which had to be all (or almost) included in the photo.
This also helps to have a lot of contrast between the leaf and the background, thus avoiding potential photos on green backgrounds, which would make it extremely critical to separate the leaf from the background.
The photo must also have a good perspective, avoiding viewpoints that distort the perpendicularity of the sheet's sides.

To distinguish the height from the width of the leaf, we ask that the vertical axis of the leaf be aligned with that of the sheet, and that the leaf has the tip oriented towards the top of the photo.

We also ask that the leaf be fairly flattened, to avoid shadows on the sheet: these, having a green halo, would risk confusing the algorithms into thinking that the leaf is wider than it actually is.

Finally, it is necessary that the background behind the sheet is dark, with enough contrast, to allow the algorithm to correctly identify the sheet, separating it from the background.
If the background causes this algorithm to malfunction, there is a risk that the reference measurements are incorrect, or that subsequent steps consider areas outside the sheet as "part of the leaf".

## 4. Dataset

### 4.1. Leaves

As previously mentioned, the leaves that make up the dataset are taken from plants in our gardens (with the Italian names):

-   **Acero rubrum**: a green variant of the more well-known red maple
-   **Agrifoglio**: the typical Christmas plant, but in our case without thorns, or with very minute thorns
-   **Faggio**: an oval leaf of medium size, typical of mid/high mountain areas
-   **Frassino**: very long, dark, and pointed, with serrated edges
-   **Gaggia**: with light green, small, and rounded leaves
-   **Liquidambar**: a large tree with huge five-pointed leaves
-   **Mirtillo domestico**: medium-sized, very shiny, and fairly pointed leaf
-   **Salvia**: a plant with slightly hairy leaves, very light in color
-   **Ulivo**: extremely elongated, very shiny leaves

### 4.2. Photographs

For each plant, we took 11 photographs, 10 of which made up the [training set](dataset/images/), while the remaining one was used as [test set](testset/).

The photos were taken with our phones, setting the highest possible resolution, and were not taken in an ideal environment.
The lighting conditions are very variable; some images are underexposed while others are overexposed, and there are sometimes shadows or reflections, both on the leaves and on the paper and background.
The sheet sometimes has a slight trapezoidal perspective despite the project specifications trying to avoid it.
In general, the images are very similar to the photos that a potential user of our system might take in a home environment with common means.

### 4.3. Values Calculation

From the images, the features indicated in the following paragraphs are extracted, then aggregated by plant, and finally discretized to calculate the probabilities used by the classifier.

During the calculation, there is a caching mechanism to avoid the same algorithm being executed multiple times to calculate the same value when it is a prerequisite for multiple other algorithms.

At the end of the leaf processing, the values related to each leaf are saved in the corresponding JSON file, so they are considered already calculated at the next dataset update.
An update requires heavy calculations only when a new image is added (and it is all to be processed), or a new feature is added (and it must be added to each image).

At each update, the discretizations of the features and the classifier probabilities are recalculated from scratch, as the calculation is not too onerous and could change considerably even with small variations in the training set.

## 5. Preprocessing

### 5.1. Pixel Dimensions

The dimensions of the leaves are initially measured in pixels but must then be converted into a unit of measure independent of the camera-leaf distance.
To do this, we chose to transform them into mm, using the information that the A4 sheet has dimensions of 210×297 mm.

An algorithm considers the photo in 5 different segments (both on the horizontal and vertical axes), in a range between 40% and 60% of the width/height of the image.
For each segment, it calculates the number of pixels of the sheet in that row/column.
The "final" value is obtained as the median of these five and is related to the corresponding dimension of the A4 sheet, to find how many mm each pixel is on average.

### 5.2. Paper ROI

To effectively work on the leaf, it is necessary to eliminate unnecessary variability, in our case the background behind the sheet.
Therefore, a region of interest is extracted that includes only the leaf and the white sheet of paper.

Initially, an estimate of the positions of the sheet's sides is found, working on a thresholding of the image in HSL.
The estimates are calculated as the median of 30 sheet-edge distances for each side.

The Hough transform of the found sheet edge is then calculated, limiting itself to recognizing vertical or horizontal segments.
The segments are then categorized into the 4 sides, recording for each side the extreme with the most internal position.
The final ROI is a conservative version of what was found here, to ensure it does not contain pixels outside the sheet.

### 5.3. Leaf Mask

Some feature extraction algorithms need a mask indicating where the leaf is and where the sheet background is.
This is done through a thresholding on the three HSV channels: checking that the hue is in the green tones, that the saturation is high enough not to be white, and that the value is not too high to be black.

## 6. Extracted Features

### 6.1. Height

Initially, we thought of using a quadratic approach to simultaneously identify the height, width, and shape of the leaf.
However, this proved to be computationally too expensive to be completed in acceptable times.
For this reason, the height calculation was transformed using a recursive dichotomous approach to separately find the highest and lowest points of the leaf.

### 6.2. Width at Different Levels

To identify the shape of the leaf, we measured its width at 11 equidistant levels, every 10% of height.
This then proved too impactful on the Bayesian classifier, so the levels were reduced to 6 (0%, 20%, 40%, 60%, 80%, 100%).

### 6.3. Maximum Width

We also wanted to describe the maximum width of the leaf, not in terms of physical width, but as the width of the smallest rectangle that completely contains the leaf.

To do this, we started from the widths at different levels calculated previously, from which we found a candidate as the "leftmost point." From here, an "exploration" of the leaf mask starts to go as far left as possible.
When this path is blocked, the algorithm has found the leftmost point of the leaf.
It is then repeated in a specular way to identify the rightmost point.

### 6.4. Average Color

Given the mask calculated in pre-processing, the average values of H, S, and V are calculated on the relative pixels of the image to get an indication of the leaf's color.
This helps, for example, to distinguish _salvia_, very light, from _ulivo_, extremely dark.

### 6.5. Tip Angle

From the leaf mask, the edge is extracted using the gradient, then the probabilistic Hough transform identifies the many short segments that make up the edge.
A small number is then extracted starting from those in the highest position (i.e., the position where the leaf has its tip); these segments are examined starting from the highest one, and the angle between the two that identify the tip is found, choosing them by comparing the relative position and the angular coefficient.

### 6.6. Contours

Starting from the leaf mask, an important filling operation is performed using a large circular kernel, followed by a noise cleaning operation and an image enlargement (a thin border of black pixels added all around) so that leaves that exceed the sheet dimensions are recognizable as closed contours.
This first phase is fundamental to obtaining a mask without "holes" inside that could deceive subsequent functions, but it has the side effect of rounding the tips and slightly filling the deepest concavities, effects that do not, however, prejudice the validity of the subsequently calculated features.
The OpenCV contours functions are applied to the resulting image to obtain the outer contour of the leaf (if more than one contour is detected, the one enclosing the largest pixel area is kept).

#### 6.6.1. Perimeter

The length of the extracted contour, mainly influenced by the size but also by how smooth (shorter contour) or jagged the leaf edge is.

#### 6.6.2. Concavity

Quantifying concavity is essential to distinguish leaves with more tips (like _acero rubrum_ or _liquidambar_) from others.
This quantity is calculated for each leaf as the difference between the leaf area and the convex hull area.
Note that this feature is also partly influenced by the type of leaf edge, more or less jagged.

## 7. Classifier

### 7.1. Bayesian Classifier

Due to the discrete nature of the Bayesian classifier, the features are discretized during the dataset update phase: the algorithm chooses the best number of "discretization bins" among the most common choices:

-   10
-   max(1, ⌊2 · log10(len(data))⌋), which in our case (with 90 data points) is 3
-   ⌊1 + log2(len(data))⌋, which in our case is 7
-   ⌊√len(data)⌋, which in our case is 9

This number and the discretization thresholds are saved to discretize new data to be classified in the same way.

### 7.2. Probability Calculation

The Bayesian classifier is based on the probability P(C) that a generic data point has class C, and the probability P(Xi = xi|C) that a data point has feature Xi with value xi, knowing that it has class C.

During the dataset update phase, the various P(C) and P(Xi = xi|C) are calculated and saved in a file containing all the dataset information.
For the calculation of the values P(Xi = xi|C), the calculation is slightly modified:

-   It could happen that a new image has a feature that falls into a "discretization bin" where there are no values for the true class of the test set.
    This would completely zero the probability of classifying the image with that class.
    To avoid this, each bin has a default fictitious element always present.
-   To avoid this having too high an impact (there could be up to 10 fake data for only 10 real data), each data point related to an image counts with a weight of 3 (as if there were 3 identical images).

### 7.3. Results

The classifier correctly classifies all the images in the test set, with an estimated accuracy between 81.6% and approximately 100%.

## 8. Usage Instructions

All the necessary libraries for use can be downloaded with the command `pip install -r requirements.txt`.
It is recommended to work in a virtual environment to avoid conflicts.

The program can be launched from the command line as `python ./main.py`, if you are in the main project folder with the terminal.

The program offers the following commands:

-   `python ./main.py update`: Updates the JSON files and classifier probabilities with any new images and/or features.
-   `python ./main.py classify --img <path>`: Classifies the image located at `<path>`.
    Adding the `--verbose` option provides the probabilities for all classes.
-   `python ./main.py classify --dir <path>`: Classifies all the images inside the folder at `<path>`.
    Adding the `--verbose` option provides the probabilities for all classes.
-   `python ./main.py rmfeature --feature <name>`: Removes the classifier feature named `<name>` from the JSON files that maintain the cache of the training set images.
-   `python ./main.py rmfeature --internal <name>`: Removes the internal program feature named `<name>` from the JSON files that maintain the cache of the training set images.
-   `python ./main.py correlation`: Displays a correlation matrix between the various features to verify the assumption of the naive Bayesian classifier.
Adding the `--abs` option shows the same matrix but with the absolute value of the correlation.

## 9. Improvement Suggestions

While we are fully satisfied with the result obtained, we know that anything can be improved and is far from perfect.
There are indeed some aspects of the project that could be improved.

The first point does not concern codes or algorithms but the dataset.
Being a small project developed at home, we collected an extremely small and not very high-quality dataset.
Therefore, a significant improvement could be to substantially expand the dataset, both by adding more species and by adding more photos, perhaps taken in more controlled conditions and with better quality tools.

The second improvement could involve the sheet recognition system and the initial region of interest: it would be interesting to be able to evaluate photos in which the sheet is presented slightly askew and/or the photo distorts its shape by introducing a perspective.
For example, Harris corner detection could be used to identify the corners and then perspectively transform the sheet to make it a perfect rectangle, respecting the original dimensions, to have a more precise pixel size.

A third improvement would certainly be related to execution time: our algorithms are quite slow, definitely unsuitable for real-time contexts.
It would be interesting to improve the feature algorithms or replace the features with others that achieve similar results but in less time.

One last improvement that comes to mind would be to implement an additional feature that recognizes the leaf veins (evaluating their number, length, general shape, and relative...), perhaps using some type of skeletonization.
We conducted tests implementing skeletonization algorithms both "by hand" and by using extended OpenCV libraries, but the results were not deemed satisfactory.
The problems detected were mainly caused by images in which reflections on the leaf surface made it extremely difficult to distinguish the veins even by eye, and by the enormous variability between species, some with quite recognizable veins and others extremely problematic.
