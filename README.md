# Features and Calculation for Summarization Datasets


## Note:
If this is useful for your project, please cite these two papers:
* [CDEvalSumm: An Empirical Study of Cross-Dataset Evaluation
for Neural Summarization Systems](https://arxiv.org/pdf/2010.05139.pdf) (See section 4.3)
* [DataLab: A Platform for Data Analysis and Intervention](https://arxiv.org/pdf/2202.12875.pdf) (See [1](https://expressai.github.io/DataLab/docs/WebUI/compare_two_datasets) [2](https://datalab.nlpedia.ai/normal_dataset/6176883933e51a7edda9dd68/dataset_featurize))


## 1 Install DataLab
```
pip install --upgrade pip
pip install datalabs
python -m nltk.downloader omw-1.4
```


## 2. Example
```
python example.py
```


print(dataset["train"][0])
```JSON

{'texts': ['Residual disease after initial surgery for ovarian cancer is the strongest prognostic factor for survival. However, the extent of surgical resection required to achieve optimal cytoreduction is controversial. Our goal was to estimate the effect of aggressive surgical resection on ovarian cancer patient survival.\n                A retrospective cohort study of consecutive patients with International Federation of Gynecology and Obstetrics stage IIIC ovarian cancer undergoing primary surgery was conducted between January 1, 1994, and December 31, 1998. The main outcome measures were residual disease after cytoreduction, frequency of radical surgical resection, and 5-year disease-specific survival.\n                The study comprised 194 patients, including 144 with carcinomatosis. The mean patient age and follow-up time were 64.4 and 3.5 years, respectively. After surgery, 131 (67.5%) of the 194 patients had less than 1 cm of residual disease (definition of optimal cytoreduction). Considering all patients, residual disease was the only independent predictor of survival; the need to perform radical procedures to achieve optimal cytoreduction was not associated with a decrease in survival. For the subgroup of patients with carcinomatosis, residual disease and the performance of radical surgical procedures were the only independent predictors. Disease-specific survival was markedly improved for patients with carcinomatosis operated on by surgeons who most frequently used radical procedures compared with those least likely to use radical procedures (44% versus 17%, P < .001).\n                Overall, residual disease was the only independent predictor of survival. Minimizing residual disease through aggressive surgical resection was beneficial, especially in patients with carcinomatosis.\n                II-2.'], 'summary': 'We found only low quality evidence comparing ultra-radical and standard surgery in women with advanced ovarian cancer and carcinomatosis. The evidence suggested that ultra-radical surgery may result in better survival.\xa0 It was unclear whether there were any differences in progression-free survival, QoL and morbidity between the two groups. The cost-effectiveness of this intervention has not been investigated. We are, therefore, unable to reach definite conclusions about the relative benefits and adverse effects of the two types of surgery.\nIn order to determine the role of ultra-radical surgery in the management of advanced stage ovarian cancer, a sufficiently powered randomised controlled trial comparing ultra-radical and standard surgery or well-designed non-randomised studies would be required.', 'texts_length': 305, 'summary_length': 112, 'density': 0.5772357723577236, 'coverage': 0.44715447154471544, 'compression': 2.3008130081300813, 'repetition': 0.02702702702702703, 'novelty': 0.9572649572649573, 'copy_len': 1.1458333333333333, 'oracle_position': 6.666666666666667, 'oracle_score': 0.3251231527093596}


```
print(dataset["train"]._stat)
```JSON
{'avg_train_texts_length': 2568.440565031983,
 'avg_train_summary_length': 67.66257995735607,
 'avg_train_density': 1.1680491618671676,
 'avg_train_coverage': 0.6766106257892701,
 'avg_train_compression': 43.87020178292765,
 'avg_train_repetition': 0.015228069835287699,
 'avg_train_novelty': 0.7918251875811904,
 'avg_train_copy_len': 1.3460699766312554,
 'avg_train_oracle_position': 51.17812722103764,
 'avg_train_oracle_score': 0.3867809164380068}
```

## 3. Define and Implement sub-task dependent Feature Functions

* Question: for different sub-tasks of text summarization, how do we define the features for them?

#### Single Document Summarization
[Function](https://github.com/ExpressAI/Summverse/blob/23c389289f2f7743a9d88190a586ae927be8f170/featurize/get_summ_features.py#L23)


#### Multiple Document Summarization
[Function](https://github.com/ExpressAI/Summverse/blob/23c389289f2f7743a9d88190a586ae927be8f170/featurize/get_summ_features.py#L49)

