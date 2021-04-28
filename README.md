# bert_training
Bert training guideline.
you need to download the pre-trained BERT model
pre-trained BERT model download address : https://github.com/google-research/bert 
<img width="787" alt="image" src="https://user-images.githubusercontent.com/50358274/116375298-1e0ac280-a84a-11eb-902d-ece1f65c510d.png">

Recomended Model
  - L=12, H=768 -> BERT-base model
  - L=8, H=512 -> BERT-Medium model
  - L=4, H=512 -> BERT-Small model

Hyper parameter for training
 - batch sizes : 8, 16, 32, 64, 128
 - learning rates : 3e-4, 1e-4, 5e-5, 3e-5
