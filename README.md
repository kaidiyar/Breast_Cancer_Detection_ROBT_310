# Breast Cancer Detection ROBT 310 Final Project
Project mostly focused on convolutional neural networks and Swin Transformer baselines, which are excellent in classifying different images. Dataset was obtained from the open-source website, Hugging face, where we had almost 850 ultrasound images in three classes: benign, malignant, and normal. Then we cleared the whole dataset from trash images, and also reduced it by combining malignant and benign classes and expell the unnecessary images, and  at the end obtained around 450 ultrasound images with 70 to 30 proportion. Moreover, we implemented some python libraries by hand, for example, custom CNN from scratch, and re-implemented Swin Transformer; however, because of difficulties with Transformer, we only partially write the code, and then used the open-source code from GitHub. Then by training more than 250 times, we obtained a decent result, and found that CNN is more likely to be correct rather than Swin Transformer, since we had only small dataset, which cannot be used to fully show the capabilities of Swin Transformer. At the end, we tried to use Multimodal models to describe the status of lump, but because of the limitation in resources, unrealistically to do.
---
## Datasets
[Vanilla_Dataset](https://drive.google.com/drive/folders/1xc5ZUS-f4m5oSeeTq4zzZWtCN53FGsAU?usp=drive_link)
[PARULA_Dataset](https://drive.google.com/drive/folders/12DxP1nV5omzJ7bNNmpZ1mG3e7kZgcBw1?usp=drive_link)

## Video
[VIDEO](https://drive.google.com/drive/folders/1stBRxft3-Zfud9iaY7rO5R5WqF4AqOeM?usp=drive_link)
## Libraries
- Python
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
- Image Processing
  - PIL  
- Metrics
  - Scikit-learn
- Deep Learning  
  - PyTorch
  - Tensorflow   
