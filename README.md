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

This is implemented by the deep learning model, of which main program (**<u>biopeptide_extraction.py</u>**) is deposited in the root path.

### 2.2 prioritization of oligopeptide candidates

We designed the indicators of *Enrichment significance* and *Function score* to jointly prioritize oligopeptide candidates, of which main program (**<u>enrichment_analysis.py</u>**) is deposited in the root path.

- Candidates with low enrichment significance or poor function scores, indicating a reduced likelihood of possessing the desired biological activity, are deprioritized for further validation.
- In practice, we recommend researchers to select the top *N* candidates based on the function score for wet lab verification, depending on the throughput of their laboratory platform.



# Contact

We encourage users to reach out xbaichuan95@gmail.com directly if specific clarifications or assistance are needed.

