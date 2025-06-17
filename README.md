# Tissue-classification
# Deep convolutional neural network: ResNet-26 and Attention network
 
Architecture of ResNet-26 and the attention network (1,2). An input whole slide H&E image goes through five major modules: a feature extractor, an attention weight MLP, an instance embeddings MLP, pooling step, and slide-level classification. In ResNet-26 the identity shortcuts can be directly used when the input and output are of the same dimensions (solid line shortcuts) or when the dimensions increase (dotted line shortcuts). An early layers able to encode primary features, a mid layers can encode intermediate features, and the late layers can encode advanced and more complex features.

<img width="1500" alt="image" src="https://github.com/user-attachments/assets/cf657c35-0524-4d1a-bf1f-6a67cefada59" />

Reference:
1. Maximilian Ilse, Jakub Tomczak, Max Welling. Attention-based Deep Multiple Instance Learning. Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2127-2136, 2018.
2. Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.



# Recurrence and treatment effect classification directly from histopathology in glioblastoma 

Abstract 

Glioblastoma is a primary CNS malignancy with high mortality and morbidity. While resection and adjuvant chemoradiotherapy remain the gold-standard, this infiltrative malignancy invariably recurs and patients may therefore require additional operative intervention and/or investigational treatment. In cases of the former, differentiating of true tumor recurrence from treatment effect is a central challenge in neuro-oncology. These very different outcomes often appear similarly and may even manifest with similar clinical symptoms, confounding physicians in that difficult process. Yet, there are no established quantitative histopathological parameters for the distinction between bona fide recurrent tumor and treatment effect which is crucial for diagnosis, treatment planning and clinical trials. To address this need, we developed an attention deep learning network architecture to classify recurrent glioma, treatment effect and brain tissue directly from 200 whole-slide high-resolution H&E-stained surgical specimens. We also sample the decision-making tiles of the attention deep learning network (attention weights) as a spatial map of histopathological annotation illustrate the patterns that enabled successful predictions. H&E- specimens were pre-characterized with immunohistochemical stains (IHC) for tumor burden (SOX2), microglial and monocytic inflammation (CD68), proliferation (KI67) and neurons (NeuN). Based on these stains, biopsies were categorized into recurrent glioma, treatment effect and brain tissue using k-means clustering analysis, which was then used as a ground truth for the classifier. The deep learning classification performance was 85% accurate (95% CI 84%-86%). The sampled spatial maps of histopathological annotation corresponded to the SOX2, CD68, KI67 and NeuN expressing regions in serial sections, even though this information was not used by the attention deep learning network. Thus, since the proposed ADLN used solely H&E stained slides, the need for extensive IHC staining and manual interpretation is reduced. In addition, the sampled learning of tile-based attention showing spatial histopathological patterns associated with IHC information consistent with known biological information. Our attention deep learning network accurately distinguished between recurrent glioma, treatment effect and infiltrated brain, and identify histopathologically relevant tissue-level patterns which can be further validated by the colocalization with the relevant IHC stains. This model can provide an objective means to differentiate recurrent tumor in the setting of treatment-induced tissue effects, and thus addresses a major challenge in the assessment of therapeutic response in neurooncology.



<img width="316" alt="image" src="https://github.com/user-attachments/assets/afc565fa-adbe-4ab5-891b-2c38318f7cf6" />

Classification Results for the Validation Set. The performance was assessed on the basis of accuracy, precision, F1-Score and recall.




![image](https://github.com/user-attachments/assets/2f50f0aa-4651-427e-a2af-1c99ddf51ea3)

Decision-making annotations shown in red implies that the location is most important for prediction, a blue patch represents the location that is least important for prediction. 





Method

The attention deep learning network comprised of five major modules: a feature extractor, an attention weight multi-layer perception (MLP), an instance embeddings MLP, pooling step, and slide-level classification. For feature extraction from the RBG image input, the basic residual block from the family of Resnet architectures was used which turn the network into its counterpart residual version. Shortcut connections were inserted that perform identity mapping, and their outputs were added to the outputs of the stacked layers. This block was re-implemented without batch normalization layers in the ResNet-26 configuration with 4 macro residual blocks outputting each 64, 128, 256 and 512 feature channels. As well as step the number of channels in the four layers according to [20, 40, 60, 80]. Each macro block consisted of 3 residual blocks. Each residual block had two convolutional layers consisting of 3 × 3 convolutional filters. The early layers in the convolutional neural network were used for detecting lower level features while later layers were used for detecting complex features. We  used multiple GPUs, such that four random configurations of instances were normalized separately. The output of the fully connected layer was a 2D tensor Fni, with N instances (n) with I features (i) per instance by passing input tile Tn into the feature extractor.

![image](https://github.com/user-attachments/assets/e6d14e77-40b2-456b-b0ec-a729a6d4965e)

Following feature extraction, the tile features were collated on single GPU. To regularize the ResNet optimization, the Kullback–Leibler divergence of the per tile feature distribution was calculated to that of a unit Gaussian which was included in the loss function. 

The raw attention weights were generated with a two-layer multi-layer perception (MLP). We enabled mixing between the attention-based and average-based pooling approaches through the introduction of learn-able bias δc. This bias is applied to each attention layer separately and is scaled via the sigmoid function σ as well as freely altered during training to favor the attention-based mapping. See proof at the supplemental information that optimization of the attention weights implies a relative performance gain from the average pooling based approach. The activation function for the attentions weights, softplus, was chosen. We include a special kind of batchnorm that acts to normalize the slide features, we call slide norm, SN. 

![image](https://github.com/user-attachments/assets/32847e05-c281-4dcd-b523-5ddff5b7f3a1)

The instance embeddings were generated with MLP with a similar structure. Dropout was implemented after the relu and softplus activation (preserve sufficient weight gradients) to mask both features and whole instances randomly. The hidden layer, h, the number of attention maps c, and the instance embedding b and Uhi, Vch, Xhi, Ybh are trainable parameters:

![image](https://github.com/user-attachments/assets/357acb13-38c4-4838-8c76-b1b488326194)

The whole-slide embedding was calculated by summing the instance embeddings multiplied by the unit-normalized attention weights for each layer.

![image](https://github.com/user-attachments/assets/4ab6ff0a-3574-483b-8811-0843f4031466)
 
Finally, the slide embedding M is passed through a final fully connected layer to obtain 3 logits, which define the overall slide classification as either recurrent glioma, treatment effect and brain tissue . The model is depicted in Figure 1.
The network is trained using the ADAM optimizer (computes adaptive learning rates for each parameter) with a learning rate of 0.0002, β1 = 0.9, β2= 0.99 on the combined loss function with w1 = 0.001, w2= 0.01, and the use of cross-entropy with label smoothing:

![image](https://github.com/user-attachments/assets/2b9678ad-51b0-4678-85d7-6259143a902e)

