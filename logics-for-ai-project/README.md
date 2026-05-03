# Development of a classifier to predict GAD in depressed subjects: a Gender-Bias assessment

## Abstract
Depression, often referred to as Major Depressive Disorder (MDD), is one of the most widespread mental disorders, especially among adults. It exhibits high comorbidity with several anxiety disorders, including Generalised Anxiety Disorder (GAD). 
Furthermore, evidence from the literature suggests that women are more prone to be diagnosed with GAD relative to men, probably due to the different ways in which the two sexes exhibit clinical symptomatology, with men tending to show more externalizing symptoms (such as anger, aggression or risky behaviors), making it difficult for them to be correclty diagnosed through the use of traditional diagnostic methods.
Aim of this project is to develop a classifier to predict GAD in depressed individuals, and to evaluate the possible presence of gender bias in the model's performance, by implementing the formal concepts of Group Fairness and Individual Fairness. 

## Introduction
Depression and anxiety disorders are two of the most frequent mental disorders, mainly diagnosed during adulthood. Subjects with depressive symptoms are often diagnosed with anxiety disorders, and vice versa, confirming a high comorbidity between depression and anxiety. Moreover, epidemiological studies reveal that the prevalence of depression and anxiety in women is much higher compared to that of men (WHO, 2017). This phenomenon might be explained by a greater exposure of women to negative life events (such as gender discrimination or sexual abuses), or by the biological differences in response to stress (Hasin et al., 2018). However, this aspect might also be related to the different ways of expressing the disorder for males and females: men often show more externalizing symptoms (e.g., anger, aggression, risky behaviors), in contrast to women who tend to express a more typical and internalizing symptomatology, thus more likely to be captured by traditional diagnostic tools and criteria (Walther, 2025). 

The different prevalence of depression and anxiety between male and female patients might be a cause of a gender-biased clinical diagnosis. In this context, the main purpose of the current project is to merge Machine Learning techniques and clinical knowledge in order to develop a classifier that, after being trained on a real and large dataset, is able to predict GAD in patients with significant depressive symptoms. This will then allow us to apply the formal concepts of Group Fairness and Individual Fairness, which are necessary to verify whether the model is gender-biased in its predictions.


## Theoretical Framework

### Formal definitions
The developed Python code and the Gender-Bias assessment of the classifier entirely rely on the framework provided by Manganini & Primiero (2024). The following formal definitions are adopted:

* PR = {gender} is the protected attribute, i.e., the variable that is not supposed to affect the model's output
* NPR = {phq_score, pss_score, isi_score, age, education, smoke, drink} is the set of non-protected attributes, i.e., the set of features in the dataset that can legitimately affect the decision of the model
* T = {GAD_binary} is the binary target attribute

The Training Sample S<sup>train</sup> = {D<sup>train</sup>, P<sup>train</sup>} is such that:
* D<sup>train</sup> = the 80% partition of df_clean, including only subjects with depressive symptoms and aged between 16 and 80 years
* P<sup>train</sup> = PR ∪ NPR ∪ T

The Test Sample S<sup>test</sup> = {D<sup>test</sup>, P<sup>test</sup>, w} is such that:
* D<sup>test</sup> = the remaining 20% partition of df_clean, used to evaluate the classifier's predictions
* P<sup>test</sup> = PR ∪ NPR
* w = the coefficient assigned by the Logistic Regression model to each feature, representing its relative contribution to the prediction of GAD_binary. Features with negligible coefficients 
are assigned w = 0 and excluded from the final model

### Group Fairness
 The assessment of Group Fairness (i.e., whether the model is equally fair for both males and females) is carried out by first computing the Correction Distance, which formally measures how far the model is from correcting its prediction toward the ground truth. The formula is the following:
 
 C(S, ψ) = 1 - (N - M) * accuracy

 where:
 * N = the total normalized weight available to the model. In our case, N = 1 since all feature weights are normalized
 * M = the probability already assigned by the model to the correct class. If the true label is GAD = 1, then M = pred_proba; if the true label is GAD = 0, then M = 1 - pred_proba. In both cases, (N - M) represents how far the model is from full confidence in the correct class
 * accuracy = the observed overall accuracy of the final model  

Once the Correction Distance for each subject is calculated, Group Fairness is evaluated as follows: 

Fairness<sub>group</sub>(S,T) = | C(S, GAD(a)) − C(S, GAD(b)) | < ε

where a denotes a female patient and b denotes a male patient. Group Fairness is satisfied if the absolute difference between the mean Correction Distance of females and that of males is smaller than the threshold ε, set to 0.05 following standard practice in the literature.

### Individual Fairness
The assessment of Individual Fairness (i.e., whether two similar individuals only differing by gender are equally treated by the model) first requires the computation of the Jaccard Index, which measures the similarity between the two set of features of two individuals, A and B, defined as the ratio between the size of their intersection and the size of their union::

J(A, B) = |A ∩ B| / |A ∪ B|

According to Manganini & Primiero (2024), two subjects are defined blindly similar if and only if:

* They only differ by gender, namely, the only variable they are not sharing is the protected attribute "gender".

* The Jaccard Index is higher than 0.5, ensuring a substantial overlap between their NPR features profiles.

Once blindly similar pairs are identified, Individual Fairness is evaluated as follows:

Fairness<sub>ind</sub>(S,T) = | C(S,GAD(a)) − C(S,GAD(b)) | < ε

where, for any male-female pair with Jaccard > 0.5, the formula verifies whether the absolute difference between their Correction Distance is less than a threshold ε, meaning that the model is individually fair for that specific pair. 
 

## Methodology

### Dataset
For the purpose of this study, the ideal dataset should obviously report the gender of all the subjects and should contain clinical features allowing to identify the presence of depression and/or anxiety.    
The chosen dataset is a real dataset from the paper "Temporal dynamics in psychological assessments: a novel dataset with scales and response times" (Zhao, 2023). It comprises data from 24,292 students who completed four recognized psychological scales: PHQ-9 (depression), GAD-7 (anxiety), ISI (insomnia), and PSS (perceived stress), in addition to demographic variables such as gender, age, education and smoking / drinking habits. 
The dataset includes the specific answers of the subjects and the response time to each item of the clinical questionnaires, but for the purpose of the project these features were excluded from the analysis. Only the final score for each clinical scale was considered. Further information about the dataset and a preview of each file is available at the following link: https://zenodo.org/records/10423537

### Logistic Regression
Logistic Regression is the chosen learning function for the current model. It is perfect to address the classification task of the project because, based on data in input, this algorithm outputs probabilities between 0 and 1, and applies a threshold (e.g., 0.5) to make the final prediction. In our scenario, given the set of clinical and demographic features in input, the model will predict whether the subject has a Generalized Anxiety Disorder (GAD = 1) or not (GAD = 0).

### Data preparation
Since the project is intended to detect the possibility of GAD in subjects who are already exhibiting a significant depressive symptomatology, one important step regards selecting only subjects with a final PHQ-9 score >= 10. The PHQ-9 is a brief self-report scale specific to diagnose depression and determine its severity according to the criteria of the Diagnostic and Statistical Manual of Mental Disorders (DSM-5). The PHQ-9 score has a range from 0 to 27, but a score of 10 is the general cut-off for clinically relevant depression (Gilbody et al., 2007).  

A further step involves restricting the sample to subjects with an age between 16 and 80 years. Indeed, an inspection of the dataset revealed extreme age values (e.g., 1 or 99 years), which clearly represent reporting errors. Although excluding individuals with a certain age will reduce the final sample, such an issue cannot be ignored because 'age' might constitute a relevant property within this context. The mean age in the dataset is 20.65 years, making such extreme values implausible. Since all participants are supposed to be University students, subjects younger than 16 or older than 80 have been removed to prevent these outliers from affecting the performance of the model. 

Finally, in order to compare the output of the model against a ground truth, a clinical value indicating the presence of GAD is required. The target variable is therefore derived from the final score of the GAD-7 questionnaire, then converted into a binary format. The GAD-7 questionnaire is a screening tool to detect and assess the severity of GAD symptoms. A score of 10 or greater is the standard cut-off for GAD identification (Spitzer et al., 2006). Consequently, subjects with a GAD score >= 10 are assigned the binary label GAD = 1, and GAD = 0 otherwise.        

### Encoding
Categorical variables (education, smoking habits and drinking habits) were converted into ordinal numeric values before the model training, since Logistic Regression requires numeric input. Each variable was encoded following a logic order: for instance, education level ranges from 0 (associate degree) to 3 (doctorate degree), and smoking frequency from 0 (never smokes) to 3 (current smoker).
Gender was encoded as a binary variable (male = 0, female = 1).

### Feature selection  
Feature selection was performed step-by-step: starting from gender only as baseline model, the second training involved the inclusion of phq_score to observe its effect as potentially the strongest clinical predictor within the feature set. Next, features were added in meaningful blocks (clinical variables and demographic factors), and the model's accuracy and coefficients were monitored at each step. This approach allowed to observe how each additional variable contributes to 
predictive performance and whether it affects the role of gender in the model.

Once trained the model on the entire set of features, the coefficients assigned to each variable were analyzed and - in accordance with evidence from the literature - irrelevant features were identified and removed from the final model. Specifically:

* phq_score (0.2272) and pss_score (0.1292) are the strongest predictors: depression severity and perceived stress clinically correlate with GAD, thus showing a significant predictive weight.

* edu_encoded (0.0136), drink_encoded (0.0319) and smoke_encoded (-0.0001) are near zero: these features will be considered clinically irrelevant for GAD prediction, and are therefore assigned w = 0.

* isi_score (0.0720) and age (0.0641) show weak values, but literature review is necessary to avoid the exclusion of factors that might potentially affect the correlation between depression and anxiety. Indeed, a cross-sectional study using both GAD-7 and ISI confirms the strong association between anxiety and insomnia (Choueiry et al., 2016), while a systematic review observed a high prevalence of depression and anxiety symptoms among college students (Li et al., 2022): these findings support the idea of including isi_score and age in the final model.   

## Results
The overall accuracy of the final model is 75.57%, consistent with the accuracy of previous models (Model 4 = 75.95%; Model 3 = 75.95%). No significant difference in per-gender accuracy of the final model was observed (accuracy for females = 78.26%; accuracy for males = 80.59%).
Gender coefficient was found to be a positive value in the baseline model (0.0329) and in Model 2 (0.2283), then turned into negative and close to zero in all the following models (-0.0594 in Model 3; -0.0381 in Model 4; -0.0566 in the Final Model).  

Group Fairness was assessed by first computing the mean Correction Distance for males and females separately. The mean Correction Distance for females is 0.7640 and for males is 0.7918, with a resulting absolute difference of 0.0278: strictly below the defined threshold ε = 0.05. Thus, Group Fairness is satisfied: the classifier equally treats male and female patients at the population level. 

Individual Fairness was assessed by comparing 2500 male-female pairs (50 females × 50 males, randomly sampled). Of these, 611 pairs were identified as blindly similar (Jaccard > 0.5), and Individual Fairness was satisfied in only 145 out of 611 pairs (23.7%). The mean CD difference among all blindly similar pairs is 0.1994, confirming that the distance between correction distances of similar male and female patients is substantially larger than the threshold. Individual Fairness is therefore violated: in 76.3% of blindly similar pairs, the classifier assigns significantly different correction distances to male and female patients with equivalent clinical profiles.
In particular, in 65.5% of pairs violating Individual Fairness (305/466), the model assigns a lower correction distance to the female patient, indicating higher predictive confidence for females. This asymmetry confirms that the Individual Fairness violation is not random but it is systematically favoring female patients in GAD prediction.

## Discussion
A classifier was trained on a large real dataset using a Logistic Regression function to predict the possibility of GAD in subjects showing clinically relevant depressive symptoms, and a gender-bias assessment was carried out through the implementation of Group Fairness and Individual Fairness. 
The dataset was filtered to include only subjects with depressive symptoms (PHQ score >= 10) and aged between 16 and 80 years. Out of 24,292 participants in the initial dataset, a final sample of 1310 subjects was used to conduct the project.    

Features were added step-by-step, starting from a baseline model (gender only). Overall accuracy of the model, per-gender accuracy and features coefficients were monitored at each step. Interesting results emerged from the training of Model 2, where gender gained notable predictive weight (gender_encoded = 0.2283). However, better interpretation of this value was possible after examining Model 3, where gender coefficient went negative and close to zero (gender_encoded = -0.0594), suggesting that gender was previously acting as a proxy by carrying hidden information about other variables (such as PSS and ISI). Once directly added pss_score and isi_score in Model 3, gender lost that proxy role and its actual independent contribution became negligible. This represents a concrete example of how protected attributes introduce bias in classifiers. 

Group Fairness was satisfied (|CD_female - CD_male| = 0.0278 < ε = 0.05), indicating that the mean Correction Distance is comparable across male and female patients: the classifier is not disatvantaging a specific group. However, this result alone is insufficient to conclude that the model is unbiased.

Individual Fairness was clearly violated: the classifier was found to systematically assign higher predictive confidence to females than to males with similar clinical profiles, thus reproducing the known epidemiological tendency to associate anxiety disorders more readily with women. The discrepancy between Group and Individual Fairness represents the central theoretical contribution of this project: a classifier satisfying Group Fairness at the population level may anyway introduce systematic gender bias at the individual level.

This project shows crucial implications for the possible use of Machine Learning classifiers in the clinical field: such tools may offer a valuable support in the diagnosis of mental disorders, especially in patients who already exhibit a specific symptomatology, and might help observing correlations between multiple and different factors. However, it is equally important to recognize that these models might reproduce the same biases observed in traditional human diagnosis, thus without really improving human performance. Accuracy alone is not enough to consider a classifier valid, fair and reliable for clinical use: fairness metrics should be adopted as integral components in the evaluation of a model.  

## Limitations
Several limitations of the  project should be acknowledged.

The filtered sample shows a notable gender imbalance (805 females vs 505 males). This disproportion may affect the reliability of the fairness analysis, as conclusions 
drawn from a larger female subsample may not equally generalize to male patients.

The dataset consists exclusively of Chinese university students, with a mean age of approximately 20 years. This limits the generalization of the findings to broader clinical populations, including older adults, non-student populations, and individuals from different cultural backgrounds.

The Correction Distance was implemented using the predicted probability of the classifier as an index for the parameter M in the formal definition of Manganini & Primiero (2024). For the computations, M was set to pred_proba when the true label is GAD = 1, and to 1 - pred_proba when the true label is GAD = 0. This approximation is coherent with the framework but differs from the original definition, where M is formally expressed as a sum of feature weights, M = f(W), rather than a predicted probability.

Individual Fairness was assessed using the standard Jaccard threshold of J > 0.5. However, a stricter threshold (e.g., J > 0.7) might identify more genuinely similar pairs but at the cost of a smaller sample of comparable pairs.

The target variable GAD_binary was derived from GAD-7 scores using a standard cut-off of ≥ 10. This scale is widely validated in the literature, but it 
represents a screening instrument rather than a formal clinical diagnosis. The output of the classifier should be therefore considered as the likelihood of clinically significant anxiety symptoms, not a confirmed GAD diagnosis.


## References
Choueiry, N., Salamoun, T., Jabbour, H., El Osta, N., Hajj, A., & Rabbaa Khabbaz, L. (2016). Insomnia and relationship with anxiety in university students: a cross-sectional designed study. PloS one, 11(2), e0149643.

Depression and Other Common Mental Disorders: Global Health Estimates. Geneva: World Health Organization; 2017. Licence: CC BY-NC-SA 3.0 IGO

Gilbody S, Richards D, Barkham M. Diagnosing depression in primary care using self-completed
instruments: a UK validation of the PHQ-9 and CORE-OM. Brit J Gen Pract. 2007;57:650-652. 

Hasin, D. S., Sarvet, A. L., Meyers, J. L., Saha, T. D., Ruan, W. J., Stohl, M., & Grant, B. F. (2018). Epidemiology of adult DSM-5 major depressive disorder and its specifiers in the United States. JAMA Psychiatry, 75(4), 336–346.

Manganini, Chiara & Primiero, Giuseppe (2024). Reasoning With and About Bias. In Hykel Hosni & Juergen Landes, Perspectives on Logics for Data-driven Reasoning. Cham: Springer Nature Switzerland. pp. 127-154.

Li, W., Zhao, Z., Chen, D., Peng, Y. and Lu, Z. (2022), Prevalence and associated factors of depression and anxiety symptoms among college students: a systematic review and meta-analysis. J Child Psychol Psychiatr, 63: 1222-1230. https://doi.org/10.1111/jcpp.13606

Spitzer RL, Kroenke K, Williams JBW, Löwe B. A Brief Measure for Assessing Generalized Anxiety Disorder: The GAD-7. Arch Intern Med. 2006;166(10):1092–1097. doi:10.1001/archinte.166.10.1092

Walther A. Gender-biased diagnosis and treatment of depression: considering our blind eye on men's depression. Int J Equity Health. 2025 Jul 1;24(1):190. doi: 10.1186/s12939-025-02569-1. PMID: 40597350; PMCID: PMC12211352.

Zhao, Su. (2023). Data from: Temporal dynamics in psychological assessments: a novel dataset with scales and response times [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10423537


