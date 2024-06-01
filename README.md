# Version Information
Versions 1 is a pure implementation of Wang et al. 2010. Version 2 has code for testing with video input; in other words, locations of hands and the face are gathered from a video of a sign. We found that accuracy significantly decreases with video input, although there may be reasons for this other than imperfection of hand detectors such as a different scaling system (mediapipe reports normalized locations while the matlab files used as examples for signs are unnormalized or bugs in the code. However, the former reason, a different scaling system, is dubious to me due to the normalization of translation described in the paper. However, this normalization was created to account for minor differences in translation length due to small differences in distance from the camera; the difference between normalized values from 0.0 to 1.0 and pixel values is much is much greater than any difference that could arise from sitting further or closer to the camera.

# Future Work
Currently, I am working both on deubugging version 2 to ensure no mistakes are present in the method and creating version 3, which will include subsequence matching where videos of signs are no longer perfectly cropped, instead including parts where the signer raises and/or lowers their hand before/after performing the sign. In version 4 and 5, we plan to experiment with different models: 4 will include variations on the "vanila" DTW used in versions 1-2 to help increase accuracy while version 5 will experiement with GCNN, which have been shown to perform well on the WLASL dataset.

# Citations
Haijing Wang, Alexandra Stefan, Sajjad Moradi, Vassilis Athitsos, Carol Neidle, and Farhad Kamangar.
**A System for Large Vocabulary Sign Search.**
***Workshop on Sign, Gesture and Activity (SGA), September 2010.***
