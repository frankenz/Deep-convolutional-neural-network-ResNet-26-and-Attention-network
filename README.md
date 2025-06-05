# Deep convolutional neural network: ResNet-26 and Attention network
 
Architecture of the attention deep learning network (1,2). An input whole slide H&E image goes through five major modules: a feature extractor, an attention weight MLP, an instance embeddings MLP, pooling step, and slide-level classification. In ResNet-26 the identity shortcuts can be directly used when the input and output are of the same dimensions (solid line shortcuts) or when the dimensions increase (dotted line shortcuts). An early layers able to encode primary features, a mid layers can encode intermediate features, and the late layers can encode advanced and more complex features.

<img width="1500" alt="image" src="https://github.com/user-attachments/assets/cf657c35-0524-4d1a-bf1f-6a67cefada59" />

Reference:
1. Maximilian Ilse, Jakub Tomczak, Max Welling. Attention-based Deep Multiple Instance Learning. Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2127-2136, 2018.
2. Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
