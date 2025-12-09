<h1 align="center">
  <b> DeepMapPlus</b>
</h1>

<h3 align="center">
  <i>Deep learning meets Genomics</i><br>
  <small>An ensemble model for genotype to phenotype prediction</small>
</h3>

---

## 📝 Description

DeepMapPlus is an stacked ensemble based python package that combines deep learning with GBlup using SVR as a meta learner to estimate Genomic Estimated Breeding Values (GEBVs) and recommend optimal crosses, enabling breeders to accelerate genetic gain and streamline selection decisions in rice improvement programs.





**Epistatic Interaction**  

The model extended the standard additive framework by incorporating additive-by-additive epistatic effects, following the theoretical foundation established by Henderson (1985). This enhanced model formulation is expressed as 

**y = 1μ + Zg + Zi + ε,** 

where y represents the phenotypic observations, μ is the overall mean, Z is the design matrix, g denotes additive genetic effects [g ∼ N(0, σ²_G G)], i represents epistatic effects [i ∼ N(0, σ²_I I)], and ε is the residual error term. The epistatic relationship matrix I is constructed through the Hadamard product (element-wise multiplication) of the additive genomic relationship matrix G (I = A ⊙ A), following the standardization approach described by Vitezica et al. (2017). The inclusion of the epistatic term enables the model to capture non-additive genetic variance. The resulting model provided a computationally efficient yet biologically meaningful framework for estimating breeding values in traits influenced by epistatic interactions, while preserving the interpretability and robustness of the linear mixed model approach. 

 

**Ten-fold-cross validation** 

Cross-validation represents a standard methodology for assessing the predictive accuracy of genomic selection (GS) models (Estaghvirou et al 2013). In this study, we implemented a tenfold cross-validation scheme. For each iteration, nine subsets (90% of samples) served as the training set to develop the prediction model, while the remaining subset (10% of samples) functioned as the validation set. Following model training, phenotypic values for individuals in the test group were predicted exclusively from their genotypic data. 

 




## 📂 Input Data Requirements
**Accepted Formats:**
- Input data format is given in Example_files folder
- For generating Training and Testing set use clustering.py 
- For calculating Additive matrix use Additive_Dominance.R
##  User's Note:
- Please refer key_note_to_users file for package installation
- Use MainTuning.py for BreedSightTuning model

