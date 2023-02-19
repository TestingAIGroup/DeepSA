# Introduction of DeepSA
Surprise adequacy quantifies the neuron activation differences between a test input and the training set. It is an important metric to measure test adequacy, which has not been leveraged for test input generation. This study proposes a surprise adequacy-guided test input generation approach. Firstly, it selects important neurons that contribute more to the final decision. Activation values of these neurons are used as features to improve the surprise adequacy metric. Then, by leveraging the idea of Coverage-Guided Fuzzing Testing, the surprise adequacy value of test inputs and the prediction probability differences among classes are utilized as joint optimization objectives. Perturbations are calculated by using the gradient ascent algorithm and test inputs are generated iteratively. 

# Implementation
capture_surprise_one_lenet.py is leveraged for analyzing the accuracy of DNN models on several test suites of growing sizes.

dsa_original_lenet.py is used to measure the DSA values of the testing set.

lsa_original_lenet.py is used to measure LSA values of the testing set.

generation_util_improve_other_V2.py is utilized to generate test inputs by adopting the gradient ascent algorithm.

