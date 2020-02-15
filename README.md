# Awesome-RecSys-Works
Papers and works on Recommendation System(RecSys) you must know
### Survey Review

| Titile                                                       |             Booktitle             | Authors                                                      | Resources                                                    |
| ------------------------------------------------------------ | :-------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Deep Learning Based Recommender System: A Survey and New Perspectives | ACM Computing Surveys (CSUR)'2019 | Shuai Zhang; Lina Yao; Aixin Sun; Yi Tay                     | [[pdf]](https://arxiv.org/abs/1707.07435)                    |
| Sequential Recommender Systems: Challenges, Progress and Prospects |            IJCAI'2019             | Shoujin Wang; Liang Hu; Yan Wang; Longbing Cao; Quan Z. Sheng; Mehmet Orgun | [[pdf]](https://www.researchgate.net/publication/333044093_Sequential_Recommender_Systems_Challenges_Progress_and_Prospects) |
| Real-time Personalization using Embeddings for Search Ranking at Airbnb |             KDD'2018              | Mihajlo Grbovic (Airbnb); Haibin Cheng (Airbnb)              | [[pdf]](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) |
| Deep Neural Networks for YouTube Recommendations             |           RecSys '2016            | Paul Covington(Google);Jay Adams(Google);Emre Sargin(Google) | [[pdf]](https://ai.google/research/pubs/pub45530)            |
| The Netflix Recommender System: Algorithms, Business Value, and Innovation |           ACM TMIS'2015           | Carlos A. Gomez-Uribe(Netflix);Neil Hunt(Netflix)            | [[pdf]](https://www.academia.edu/27800721/The_Netflix_Recommender_System_Algorithms_Business_Value_and_Innovation) |

### Click-Through-Rate(CTR) Prediction

| Titile                                                       |   Booktitle    | Resources                                                    |
| ------------------------------------------------------------ | :------------: | ------------------------------------------------------------ |
| **FM**: Factorization Machines                               |   ICDM'2010    | [[pdf]](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) [[code]](https://github.com/coreylynch/pyFM) [[tffm]](https://github.com/geffy/tffm) [[fmpytorch]](https://github.com/jmhessel/fmpytorch) |
| **libFM**:  Factorization Machines with libFM                | ACM Trans'2012 | [[pdf]](https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf) [[code]](https://github.com/srendle/libfm) |
| **GBDT+LR**: Practical Lessons from Predicting Clicks on Ads at Facebook |    ADKDD'14    | [[pdf]](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) |
| **FFM**: Field-aware Factorization Machines for CTR Prediction |  RecSys'2016   | [[pdf]](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) [[code]](https://github.com/guestwalk/libffm) |
| **FNN**: Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction |   ECIR'2016    | [[pdf]](https://arxiv.org/abs/1601.02376)[[Tensorflow]](https://github.com/shenweichen/DeepCTR) |
| **PNN**: Product-based Neural Networks for User Response Prediction |   ICDM'2016    | [[pdf]](https://arxiv.org/abs/1611.00144)[[Tensorflow]](https://github.com/Atomu2014/product-nets) |
| **Wide&Deep**: Wide & Deep Learning for Recommender Systems  |   DLRS'2016    | [[pdf]](https://arxiv.org/pdf/1606.07792)[[Tensorflow]](https://github.com/tensorflow/models/tree/master/official/wide_deep)[[Blog]](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) |
| **AFM**: Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks |   IJCAI'2017   | [[pdf]](https://arxiv.org/abs/1708.04617)[[Tensorflow]](https://github.com/hexiangnan/attentional_factorization_machine) |
| **NFM**: Neural Factorization Machines for Sparse Predictive Analytics |   SIGIR'2017   | [[pdf]](https://arxiv.org/abs/1708.05027)[[Tensorflow]](https://github.com/hexiangnan/neural_factorization_machine) |
| **DeepFM**: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[C] |   IJCAI'2017   | [[pdf]](https://arxiv.org/abs/1703.04247) [[code]](https://github.com/ChenglongChen/tensorflow-DeepFM) |
| **DCN**: Deep & Cross Network for Ad Click Predictions       |   ADKDD'2017   | [[pdf]](https://arxiv.org/abs/1708.05123) [[Keras]](https://github.com/Nirvanada/Deep-and-Cross-Keras)[[Tensorflow]](https://github.com/flyzaway/Deep-Cross-Tensorflow) |
| **xDeepFM**: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems |    KDD'2018    | [[pdf]](https://arxiv.org/abs/1803.05170) [[Tensorflow]](https://github.com/Leavingseason/xDeepFM) |
| **DIN**: DIN: Deep Interest Network for Click-Through Rate Prediction |    KDD'2018    | [[pdf]](https://arxiv.org/abs/1706.06978) [[Tensorflow]](https://github.com/zhougr1993/DeepInterestNetwork) |
| **DIEN**: DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction |   AAAI'2019    | [[pdf]](https://arxiv.org/abs/1809.03672) [[Tensorflow]](https://github.com/mouna99/dien) |
| **DSIN**: Deep Session Interest Network for Click-Through Rate Prediction |   IJCAI'2019   | [[pdf]](https://arxiv.org/abs/1905.06482)[[Tensorflow]](https://github.com/shenweichen/DSIN) |
| **AutoInt**: Automatic Feature Interaction Learning via Self-Attentive Neural Networks |   CIKM'2019    | [[pdf]](https://arxiv.org/abs/1810.11921)[[Tensorflow]](https://github.com/shichence/AutoInt) |



### Sequence-based Recommendations

| Titile                                                       |     Booktitle      | Resources                                                    |
| ------------------------------------------------------------ | :----------------: | ------------------------------------------------------------ |
| **GRU4Rec**:Session-based Recommendations with Recurrent Neural Networks |     ICLR'2016      | [[pdf]](https://arxiv.org/pdf/1511.06939.pdf)[[code]](https://github.com/hidasib/GRU4Rec) |
| **DREAM**:A Dynamic Recurrent Model for Next Basket Recommendation |     SIGIR'2016     | [[pdf]](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/A%20Dynamic%20Recurrent%20Model%20for%20Next%20Basket%20Recommendation.pdf)[[code]](https://github.com/LaceyChen17/DREAM) |
| Long and Short-Term Recommendations with Recurrent Neural Networks |     UMAP’2017      | [[pdf]](http://iridia.ulb.ac.be/~rdevooght/papers/UMAP__Long_and_short_term_with_RNN.pdf)[[Theano]](https://github.com/rdevooght/sequence-based-recommendations) |
| **Time-LSTM**:What to Do Next: Modeling User Behaviors by Time-LSTM |     IJCAI'2017     | [[pdf]](http://static.ijcai.org/proceedings-2017/0504.pdf) [[code]](https://github.com/DarryO/time_lstm) |
| **Caser**:Personalized Top-N Sequential Recommendation via Convolutional Sequence EmbeddingCaser |     WSDM'2018      | [[pdf]](http://www.sfu.ca/~jiaxit/resources/wsdm18caser.pdf) [[code]](https://github.com/graytowne/caser_pytorch) |
| **SASRec**:Self-Attentive Sequential Recommendation          |     ICDM'2018      | [[pdf]](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)[[code]](https://github.com/kang205/SASRec) |
| **BERT4Rec**:BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer | ACM WOODSTOCK’2019 | [[pdf]](https://arxiv.org/abs/1904.06690)[[code]](https://github.com/FeiSun/BERT4Rec) |
| **SR-GNN**: Session-based Recommendation with Graph Neural Networks |     AAAI'2019      | [[pdf]](https://arxiv.org/pdf/1811.00855v4.pdf) [[code]](https://github.com/CRIPAC-DIG/SR-GNN) |


### Knowledge Graph

| Titile                                                       | Booktitle | Resources                                                    |
| ------------------------------------------------------------ | :-------: | ------------------------------------------------------------ |
| **RippleNet**: RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems | CIKM'2018 | [[pdf]](https://arxiv.org/abs/1803.03467)  [[code]](https://github.com/hwwang55/RippleNet) |
|                                                              |           |                                                              |

### Collaborative Filtering

| Titile                                                       |       Booktitle       | Resources                                                    |
| :----------------------------------------------------------- | :-------------------: | ------------------------------------------------------------ |
| **UBCF**:GroupLens: an open architecture for collaborative filtering of netnews |       CSCW'1994       | [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.3784&rep=rep1&type=pdf)[[code]](https://github.com/fuhailin/Memory-based-collaborative-filtering) |
| **IBCF**:Item-based collaborative filtering recommendation algorithms |       WWW'2001        | [[pdf]](http://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf)[[code]](https://github.com/fuhailin/Memory-based-collaborative-filtering) |
| **SVD**:Matrix Factorization Techniques for Recommender Systems | Journal Computer'2009 | [[pdf]](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)[[code]](https://github.com/j2kun/svd) |
| **SVD++**:Factorization meets the neighborhood: a multifaceted collaborative filtering model |       KDD'2008        | [[pdf]](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)[[code]](https://github.com/lxmly/recsyspy) |
| **PMF**: Probabilistic Matrix Factorization                  |       NIPS'2007       | [[pdf]](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)  [[code]](https://github.com/fuhailin/Probabilistic-Matrix-Factorization) |
| **CDL**:Collaborative Deep Learning for Recommender Systems  |       KDD '2015       | [[pdf]](https://arxiv.org/pdf/1409.2944.pdf)[[code]](https://github.com/akash13singh/mxnet-for-cdl)[[PPT]](http://www.wanghao.in/mis/CDL.pdf) |
| **ConvMF**:Convolutional Matrix Factorization for Document Context-Aware Recommendation |      RecSys'2016      | [[pdf]](http://dm.postech.ac.kr/~cartopy/ConvMF/)[[code]](https://github.com/cartopy/ConvMF)[[zhihu]](https://zhuanlan.zhihu.com/p/27070343)[[PPT]](http://dm.postech.ac.kr/~cartopy/ConvMF/ConvMF_RecSys16_for_public.pdf) |
| **NCF**:Neural Collaborative Filtering                       |        WWW '17        | [pdf](https://arxiv.org/pdf/1708.05031.pdf)[code](https://github.com/hexiangnan/neural_collaborative_filtering) |

### Other
DropoutNet: Addressing Cold Start in Recommender Systems. [[pdf]](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)  [[code]](https://github.com/layer6ai-labs/DropoutNet)

### Graph Neural Networks

### Transfer Learning

### Public Datasets
- **KASANDR**:KASANDR: A Large-Scale Dataset with Implicit Feedback for Recommendation (SIGIR 2017).
  [[pdf]](http://ama.liglab.fr/~sidana/PDF/SIGIR17_short) [[KASANDR Data Set ](http://archive.ics.uci.edu/ml/datasets/KASANDR)]

### Blogs
- [Deep Learning Meets Recommendation Systems by Wann-Jiun](https://blog.nycdatascience.com/student-works/deep-learning-meets-recommendation-systems/) 

- [Quora: Has there been any work on using deep learning for recommendation engines?](https://www.quora.com/Has-there-been-any-work-on-using-deep-learning-for-recommendation-engines).


### Courses & Tutorials
- Recommender Systems Specialization [Coursera](https://www.coursera.org/specializations/recommender-systems)

- Deep Learning for Recommender Systems by Balázs Hidasi. [RecSys Summer School](http://pro.unibz.it/projects/schoolrecsys17/program.html), 21-25 August, 2017, Bozen-Bolzano. [Slides](https://www.slideshare.net/balazshidasi/deep-learning-in-recommender-systems-recsys-summer-school-2017)

- Deep Learning for Recommender Systems by Alexandros	Karatzoglou and Balázs	Hidasi. RecSys2017 Tutorial. [Slides](https://www.slideshare.net/kerveros99/deep-learning-for-recommender-systems-recsys2017-tutorial)


### Recommendation Systems Engineer Skill Tree
- Skill Tree [pdf](https://github.com/fuhailin/awesome-RecSys-works/blob/master/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%B7%A5%E7%A8%8B%E5%B8%88%E6%8A%80%E8%83%BD%E6%A0%91.pdf)
