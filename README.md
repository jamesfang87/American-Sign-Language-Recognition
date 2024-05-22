# Version Information and Future Work
Versions 1-3 are purely implementations of Wang et al. 2010. Version 3 is the final version while versions 1-2 are "snapshots" of the code at certain points in implementation. For a working program, these versions should still run but are essentially useless. They are useful for the development of version 4 which is why they are still present in this repository. Upon the completion of version 4, they will be deleted and the version numbers updated: 3 -> 1, 4 -> 2, 5 -> 3, etc.
Currently, I am working on creating version 4: which extends past the work done by Wang et al. by doing "real-life" testing; instead of hand locations being given, they are detected from videos of signs. Detection of the hands is done using mediapipe. Version 5 will include subsequence matching where videos of signs are no longer perfectly cropped, instead including parts where the signer raises and/or lowers their hand before/after performing the sign.

Haijing Wang, Alexandra Stefan, Sajjad Moradi, Vassilis Athitsos, Carol Neidle, and Farhad Kamangar.
A System for Large Vocabulary Sign Search.
Workshop on Sign, Gesture and Activity (SGA), September 2010.
