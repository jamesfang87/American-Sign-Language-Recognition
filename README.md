# Version Information
Version 1 is a pure implementation of [1]. Version 2 has code for testing with video input; in other words, locations of hands and the face are gathered from a video of a sign. Version 3 includes subsequence search where videos of signs are no longer perfectly cropped. 

# Results and Analysis
We found that accuracy significantly decreases with video input (from around the correct sign being present in the top 10 predictions about 70% (consistent with the results of [1]) of the time to only around 40% of the time), although there may be reasons for this other than imperfection of hand detectors. We mainly attribute this error to potential differences in where the center of the hand is chosen to be. In our sign dictionary, hand positions are represented as the centroid of the hand bounding box. However, in our video experiments, we used mediapipe's pose detector to find hand locations, more specifically taking the location of the base of the left and right index fingers for respective hand locations. This configuration provided a 10% increase in accuracy compared to a previous configuration which used the location of the left and right wrists, highlighting the sensitivity of DTW to what exactly is used for hand location. We hypothesize that the centroid of the hand bounding box lies at a different position than the relatively central location which we chose. Interestingly, when we changed from getting the position of the center of the face from the mediapipe pose detector to the center of a face bounding box, our accuracy dropped, most likely due to noise in face detections as the size of the bounding box changes by large amounts. In addition, because of noise in hand position detections, velocity vectors which are extracted as features can be inaccurate. Accuracy also drops with the addition of subsequence search from 40% to 20%.

# Citations
[1] Haijing Wang, Alexandra Stefan, Sajjad Moradi, Vassilis Athitsos, Carol Neidle, and Farhad Kamangar.
**A System for Large Vocabulary Sign Search.**
***Workshop on Sign, Gesture and Activity (SGA), September 2010.***

[2] Vassilis Athitsos, Carol Neidle, Stan Sclaroff, Joan Nash, Alexandra Stefan, Quan Yuan, and Ashwin Thangali.
**The American Sign Language Lexicon Video Dataset.**
***IEEE Workshop on Computer Vision and Pattern Recognition for Human Communicative Behavior Analysis (CVPR4HB), June 2008.***
