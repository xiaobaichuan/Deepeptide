# **About**

Deepeptide is a general-purpose AI-based pipeline for therapeutic oligopeptides discovery on various metabolic diseases.

# **How to use Deepeptide**

Deepeptide implements its two-step pipeline to discover potent and novel oligopeptides for disease indication of interest.  In this repository, we presents source code of **the second step**, including the deep learning model and the prioritization.

## 1. **Constructing the specialized library composed of functional IDRs**

For the first step, we constructed a specialized library composed of IDRs with indication-related molecular functions.

We initiated the process with descriptive keywords of each indication. Taking the indication ‘angiogenesis’ as a case study, we detail the following processes, which were similarly applied to the other indications using their respective keywords.

### 1.1 Determining the indication-related molecular functions

- Submitted the keyword of ‘angiogenesis’ to AmiGO2 (http://amigo.geneontology.org/amigo) to retrieve the relevant biological processes. 
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

This is implemented by the deep learning model, of which main program (**<u>biopeptide_extraction.py</u>**) is deposited in the root path. There are some important files deposited in the following subdirectories:

#### 2.1.1 Data

This subdirectory includes a file named  "Training Dataset.txt", the dataset for training the deep learning model. This dataset comprises 7,028 protein-derived biopeptides mapped to 9,175 protopeptides, obtained from UniProt and publicly available therapeutic peptide databases. Each amino acid within the protopeptides was subsequently labeled according to its positional information within the parent proteins using the BIEO scheme. In this scheme, amino acids are assigned one of four tags: 'B', 'I', 'E', or 'O', indicating the amino acid is in the starting point, internal region, endpoint and outside of a biopeptide, respectively.

It is worth noting that sequence masking information plays a crucial role in determining model performance during training. This is primarily because identified biopeptides within protopeptides remain relatively scarce. When using the BIEO scheme for labeling, regions where biopeptides have not yet been discovered are typically labeled as 'O'. This approach inevitably introduces a large amount of 'false-negative' information. Therefore, incorporating appropriate sequence masking information during model training is highly beneficial for achieving a balanced model performance in terms of false positives and false negatives. However, there is no consensus on what constitutes the optimal sequence masking strategy. As a result, this type of information has not been explicitly annotated in "Training Dataset.txt".

#### 2.2.2 models

This subdirectory includes a file named  "dl_model.py", the details of the deep learning model. Specifically, the model is developed with three main components: 1) Input Representation: We fine-tuned the state-of-the-art pre-training protein large language model , ESM-2, to integrate global evolutionary information and general semantic patterns of proteins; 2) Context Encoder: A bidirectional Long Short-Term Memory (Bi-LSTM) network captures context dependencies, crucial for extracting the targeted subsequence (in this case, the oligopeptide) from the larger sequence (the protopeptide); 3) Tag Decoder: We employed a Conditional Random Field (CRF) model to assign tags to each amino acid, indicating its membership as part of the biopeptide.

#### 2.2.3 utils

This subdirectory includes three files as follows.

- "data_process.py" is a code script for preparing Training Dataset and Prediction Dataset.
- "metrics.py" is a code script for calculate performance metrics including Recall, Precision and F1-score for the deep learning model.
- "param_configs" is a configuration file that records hyperparameters of the deep learning model.

### 2.2 prioritization of oligopeptide candidates

We designed the indicators of *Enrichment significance* and *Function score* to jointly prioritize oligopeptide candidates, of which main program (**<u>enrichment_analysis.py</u>**) is deposited in the root path.

- Candidates with low enrichment significance or poor function scores, indicating a reduced likelihood of possessing the desired biological activity, are deprioritized for further validation.
- In practice, we recommend researchers to select the top *N* candidates based on the function score for wet lab verification, depending on the throughput of their laboratory platform.



# Contact

We encourage users to reach out xbaichuan95@gmail.com directly if specific clarifications or assistance are needed.

