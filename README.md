# **About**

Deepeptide is a general-purpose AI-based pipeline for therapeutic oligopeptides discovery on various metabolic diseases.

# **How to use Deepeptide**

Deepeptide implements its two-step pipeline to discover potent and novel oligopeptides for disease indication of interest.  In this repository, we presents source code of **the second step**, including the deep learning model and the prioritization.

## 1. **Constructing the specialized library composed of functional IDRs**

For the first step, we constructed a specialized library composed of IDRs with indication-related molecular functions (MFs).

We initiated the process with descriptive keywords of each indication. Taking the indication ‘angiogenesis’ as a case study, we detail the following processes, which were similarly applied to the other indications using their respective keywords.

### 1.1 Determining the indication-related molecular functions

- Submitted the keyword of ‘angiogenesis’ to AmiGO2 (http://amigo.geneontology.org/amigo) to retrieve the relevant biological processes (BPs).
- For each biological process, we retained only those functional proteins involved in promoting angiogenesis filtered in the operational panel ‘GO class’ of AmiGO2.
- We conducted a gene enrichment analysis for the retained proteins (termed as functional proteins) using DAVID v2022q1 (https://david.ncifcrf.gov/), focusing on the molecular functions precisely promoting angiogenesis.

### 1.2 Functional IDRs identification

- We first utilized IUPred2A (https://iupred2a.elte.hu/) to predict IDRs in all proteins across the protein universe.
- For each indication-related molecular function, we compiled a training dataset consisting of IDRs derived from functional proteins exhibiting the specific molecular function as the positive dataset. IDRs from an equal number of proteins unrelated to these functional proteins were randomly selected from UniProt to form the negative training dataset.
- We utilized FAIDR (https://github.com/taraneh-z/FAIDR) to build the MFF-based prediction model. 
- Using the MFF-based prediction model, we identified functional IDRs with the indication-related molecular functions within the positive and negative datasets. We employed a no-replacement sampling strategy to process all IDRs in the protein universe across multiple training epochs. For each epoch, the model was retrained with a newly constructed dataset.

## 2. Identification and prioritization of oligopeptide candidates

For the second step, we extracted the oligopeptide candidates by the deep learning model and prioritized them.

### 2.1 Identification of oligopeptide candidates

#### 2.2.1 Content

This is implemented by the deep learning model, of which main program (**<u>biopeptide_extraction.py</u>**) is deposited in the root path. There are some important files deposited in the following subdirectories:

##### (1) data 

This subdirectory includes a file named  "Training Dataset.txt", the dataset for training the deep learning model. This dataset comprises 7,028 protein-derived biopeptides mapped to 9,175 protopeptides, obtained from UniProt and publicly available therapeutic peptide databases. Each amino acid within the protopeptides was subsequently labeled according to its positional information within the parent proteins using the BIEO scheme. In this scheme, amino acids are assigned one of four tags: 'B', 'I', 'E', or 'O', indicating the amino acid is in the starting point, internal region, endpoint and outside of a biopeptide, respectively.

It is worth noting that sequence masking information plays a crucial role in determining model performance during training. This is primarily because identified biopeptides within protopeptides remain relatively scarce. When using the BIEO scheme for labeling, regions where biopeptides have not yet been discovered are typically labeled as 'O'. This approach inevitably introduces a large amount of 'false-negative' information. Therefore, incorporating appropriate sequence masking information during model training is highly beneficial for achieving a balanced model performance in terms of false positives and false negatives. However, there is no consensus on what constitutes the optimal sequence masking strategy. As a result, this type of information has not been explicitly annotated in "Training Dataset.txt".

##### (2) models

This subdirectory includes a file named  "dl_model.py", the details of the deep learning model. Specifically, the model is developed with three main components: 1) Input Representation: We fine-tuned the state-of-the-art pre-training protein large language model , ESM-2, to integrate global evolutionary information and general semantic patterns of proteins; 2) Context Encoder: A bidirectional Long Short-Term Memory (Bi-LSTM) network captures context dependencies, crucial for extracting the targeted subsequence (in this case, the oligopeptide) from the larger sequence (the protopeptide); 3) Tag Decoder: We employed a Conditional Random Field (CRF) model to assign tags to each amino acid, indicating its membership as part of the biopeptide.

##### (3) utils

This subdirectory includes three files as follows.

- "data_process.py" is a code script for preparing Training Dataset and Prediction Dataset.
- "metrics.py" is a code script for calculate performance metrics including Recall, Precision and F1-score for the deep learning model.
- "param_configs" is a configuration file that records hyperparameters of the deep learning model. 

#### 2.2.2 Input/Output formats

##### (1) Input

When train/validate the model, users should prepare an data file according to the following format:

>\>E3YBA4
>
>MTVKIAQKKVLPVIGRAAALCGSCYPCSCM
>
>OOOOOOOOOOOOOOOOOOOBIIIIIIIIIE

Clearly, each training data unit consists of three rows:
i.	The first row (starting with “>”) specifies the protopeptide’s UniProt Accession ID.
ii.	The second row provides the protopeptide’s amino acid sequence.
iii.	The third row annotates the sequence, where ‘B’ and ‘E’ indicate the start and end of cleavage sites, respectively. And, flanking sequences, representing the upstream or downstream residues adjacent to the biopeptides within the protopeptides, are annotated with the label ‘O’, distinguished from biopeptides using the cleavage sites as boundaries.

When test the model, users should prepare an data file according to the following format:

>\>Test_Sequence_0001
>
>MTVKIAQKKVLPVIGRAAALCGSCYPCSCM

Each training data unit consists of two rows, that is, sequence ID and amino acid sequence, and there is no need for providing label information in this case.

##### (2) Output

When test the model, or in other words, make inference/prediction, there is an output file to be generated, like the following format:

>\>Test_Sequence_0001
>
>MTVKIAQKKVLPVIGRAAALCGSCYPCSCM
>
>OOOOOOOOOOOOOOOOOOOBIIIIIIIIIE

Each training data unit consists of two rows, that is, sequence ID, amino acid sequence and annotation results for each amino acids.

#### 2.2.3 Usages for training/validating/test

We integrated the training, testing, and evaluating scripts in a sole script “biopeptide_extraction.py”. Users can train, validate or test the model by simply setting the mode parameter in “param_configs.py”.

##### (1) Training

Set parameter in “param_configs.py” as follows: 

```
is_training = True
is_predicting = False
```

Ensuring there is a file named "train.txt" in the subdirectories "root_path/data/train.txt",

the script “biopeptide_extraction.py” will carry out model training.

##### (2) Validation

Set parameter in “param_configs.py” as follows: 

```
is_training = False
is_predicting = False
```

Ensuring there is a file named "valid.txt" in the subdirectories "root_path/data/valid.txt",

the script “biopeptide_extraction.py” will carry out model validation.

##### (3) Test

When parameter setting in “param_configs.py” is:

```
is_training = False
is_predicting= True
```

Ensuring there is a file named "test.txt" in the subdirectories "root_path/data/test.txt",

the script turns to its test mode.

### 2.2 Prioritization of oligopeptide candidates

We designed the indicators of *Enrichment significance* and *Function score* to jointly prioritize oligopeptide candidates, of which main program (**<u>enrichment_analysis.py</u>**) is deposited in the root path.

- Candidates with low enrichment significance or poor function scores, indicating a reduced likelihood of possessing the desired biological activity, are deprioritized for further validation.
- In practice, we recommend researchers to select the top *N* candidates based on the function score for wet lab verification, depending on the throughput of their laboratory platform.

### 2.3 Minimal working example

Herein, we illustrate how to employ the deep learning model to discover angiogenic oligopeptides.

#### 2.3.1 Preparing a set of protopeptides

Protopeptides can be functional IDRs or somewhat protein sequences possessing indication-related MFs. As described in "1. Constructing the specialized library composed of functional IDRs", we can readily prepare a set of functional IDRs with indication-related MFs (demo file located in "root_path/data/Angiogenesis_positive_dataset.txt").

#### 2.3.2 Identifying biopeptides from protopeptides

As described in "2.2.3 Usages for training/validating/test", we set parameter in “param_configs.py” as follows: 

```
is_training = False
is_predicting= True
```

Run: 

```
Python biopeptide_extraction.py
```

The model will return its prediction results as the following format (saved as "prediction.txt"):

> \>Test_Sequence_0001
> MTVKIAQKKVLPVIGRAAALCGSCYPCSCM
> OOOOOOOOOOOOOOOOOOOBIIIIIIIIIE

Where the amino acids sequence annotated with "BII...IIE" represents the predicted biopeptide.

#### 2.3.3 Prioritizing biopeptide candidates

As described in section 2.2, "enrichment_analysis.py" takes:

> 1)  Peptide candidates identified by deep learning model. Demo file locates in "root_path/data/Angiogenesis_peptides.txt".
> 2)  Functional IDRs dataset prepared in 2.3.1, that is, "protopeptides.txt". Demo file locates in "root_path/data/Angiogenesis_positive_dataset".
> 3) Non functional IDRs dataset with equal size of functional IDRs dataset. Demo file locates in "root_path/data/Angiogenesis_negative_dataset_1.txt".
> 4) Protein universe means the whole proteome all over the world, and as a proof-of-concept, the protein universe in this study refers to all proteins of three species (*Homo sapiens*, *Mus musculus*, and *Rattus norvegicus*). Users are recommended to download it directly from UniProt, as size of this file exceeds the upload limitation of GitHub.

as input, run:

```
Python enrichment_analysis.py
```

The script will return the prioritized biopeptides with detailed information, as shown in the table below:

| peptide | occurrences | length  | rank_ratio | function_score |
| ------- | ----------- | ------- | ---------- | -------------- |
| PPGP    | 1439        | 4       | 0.009434   | 4.663439       |
| GPPGP   | 746         | 5       | 0.009804   | 4.624973       |
| APPP    | 1087        | 4       | 0.012579   | 4.375757       |
| ... ... | ... ...     | ... ... | ... ...    | ... ...        |

users can select the top-N peptides for subsequent wet-lab validation.

# Contact

We encourage users to reach out xbaichuan95@gmail.com directly if specific clarifications or assistance are needed.

