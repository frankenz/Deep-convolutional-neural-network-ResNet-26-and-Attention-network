# Tissue Classification in Glioblastoma 
 
Architecture of ResNet-26 and the attention network (1,2). An input whole slide H&E image goes through five major modules: a feature extractor, an attention weight MLP, an instance embeddings MLP, pooling step, and slide-level classification. In ResNet-26 the identity shortcuts can be directly used when the input and output are of the same dimensions (solid line shortcuts) or when the dimensions increase (dotted line shortcuts). An early layers able to encode primary features, a mid layers can encode intermediate features, and the late layers can encode advanced and more complex features.

<img width="1500" alt="image" src="https://github.com/user-attachments/assets/cf657c35-0524-4d1a-bf1f-6a67cefada59" />



 

There are no established quantitative histopathological parameters for the distinction between bona fide recurrent tumor and treatment effect which is crucial for diagnosis, treatment planning and clinical trials. To address this need, we developed an attention deep learning network architecture to classify recurrent glioma, treatment effect and brain tissue directly from 200 whole-slide high-resolution H&E-stained surgical specimens. We also sample the decision-making tiles of the attention deep learning network (attention weights) as a spatial map of histopathological annotation illustrate the patterns that enabled successful predictions. H&E- specimens were pre-characterized with immunohistochemical stains (IHC) for tumor burden (SOX2), microglial and monocytic inflammation (CD68), proliferation (KI67) and neurons (NeuN). Based on these stains, biopsies were categorized into recurrent glioma, treatment effect and brain tissue using k-means clustering analysis, which was then used as a ground truth for the classifier. The deep learning classification performance was 85% accurate (95% CI 84%-86%). The sampled spatial maps of histopathological annotation corresponded to the SOX2, CD68, KI67 and NeuN expressing regions in serial sections, even though this information was not used by the attention deep learning network. In addition, the sampled learning of tile-based attention showing spatial histopathological patterns associated with IHC information consistent with known biological information. Our attention deep learning network accurately distinguished between recurrent glioma, treatment effect and infiltrated brain, and identify histopathologically relevant tissue-level patterns which can be further validated by the colocalization with the relevant IHC stains. This model can provide an objective means to differentiate recurrent tumor in the setting of treatment-induced tissue effects, and thus addresses a major challenge in the assessment of therapeutic response in neurooncology.



<img width="316" alt="image" src="https://github.com/user-attachments/assets/afc565fa-adbe-4ab5-891b-2c38318f7cf6" />

Classification Results for the Validation Set. The performance was assessed on the basis of accuracy, precision, F1-Score and recall.




![image](https://github.com/user-attachments/assets/2f50f0aa-4651-427e-a2af-1c99ddf51ea3)

Decision-making annotations shown in red implies that the location is most important for prediction, a blue patch represents the location that is least important for prediction. 




References:
1. Maximilian Ilse, Jakub Tomczak, Max Welling. Attention-based Deep Multiple Instance Learning. Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2127-2136, 2018.
2. Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
   
