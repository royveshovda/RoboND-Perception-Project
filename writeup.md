## Project: Perception Pick & Place

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify).
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  See the example `output.yaml` for details on what the output should look like.  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
I followed the steps from the lectures, and copied over the implementation.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
I followed the steps from the lectures, and copied over the implementation.

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
I used the code from exercise 3 and also copied over the sensor_stick part to draw samples. I decided on 32 bins for color histogram, and also 32 bins for the normal histograms. I tried different types of classifier for the SVM, but ended back to the linear. Rbf performed well, but seems to overfit a bit. Linear gave the best combination of accuracy and generalization, at least this is my impression my after testing.

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.
I followed the instructions to get and shape all the objects. I read the pick list, and looped over all the detected objects. I decided to choose the last index from the detected objects. This was kind of a hack, based on the results I got. A better solution would be to get a confidence score from the classifier, and choose the detected object of a type with the highest confidence.


### General discussion
I decided to follow the TODO markups in the provided code template, and fill in the missing pieces. As a first step I used the sensor_stick from the exercises to draw samples from the objects in the pick lists. I decided on a rather large sample number (5000). This allowed me to detect all the objects, but the glue (I missed this one in both world 2 and 3). Further tuning and even more samples might help to get this one as well. But for a more robust perception pipeline, I would rather invest time in moving to an Deep Learning pipeline, with a Convolutional Neural Network of some kind.
