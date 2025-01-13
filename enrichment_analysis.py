import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, hypergeom
from statsmodels.stats.multitest import multipletests
from collections import Counter
import math
import seaborn as sns
import matplotlib.ticker as mtick


def count_oligopeptides_per_sequence(oligopeptides, sequences):
    counts_per_sequence = {peptide: [] for peptide in oligopeptides}
    for seq in sequences:
        for peptide in oligopeptides:
            counts_per_sequence[peptide].append(seq.count(peptide))
    return counts_per_sequence

def align_nonzero_counts(functional_counts, non_functional_counts):
    aligned_functional = [item for item in functional_counts if item > 0]
    aligned_non_functional = [item for item in non_functional_counts if item > 0]
    padding_length = abs(len(aligned_functional)-len(aligned_non_functional))
    
    if len(aligned_functional) > len(aligned_non_functional):
        aligned_non_functional += [0] * padding_length
    else:
        aligned_functional += [0] * padding_length
    return aligned_functional, aligned_non_functional

def calculate_enrichment_significance(oligopeptides, functional_counts, non_functional_counts, universe_counts, total_functional_length, total_universe_length):
    results = []
    for peptide in oligopeptides:
        functional_occurrences = functional_counts.get(peptide, [])
        non_functional_occurrences = non_functional_counts.get(peptide, [])
        functional_occurrences, non_functional_occurrences = align_nonzero_counts(functional_occurrences, non_functional_occurrences)
        if len(functional_occurrences) > 1:
            t_stat, p_value_ttest = ttest_1samp(functional_occurrences, np.mean(non_functional_occurrences), alternative='greater')
        else:
            p_value_ttest = 1

        if p_value_ttest < 0.05:
            M = sum(universe_counts.get(peptide, 0))
            n = total_functional_length
            m = sum(functional_occurrences)
            N = total_universe_length

            p_value_hypergeom = 1 - hypergeom.cdf(m - 1, N, M, n)

            results.append([peptide, p_value_ttest, p_value_hypergeom])

    results_df = pd.DataFrame(results, columns=["peptide", "p_value_ttest", "p_value_hypergeom"])

    results_df["adjusted_p_value"] = multipletests(results_df["p_value_hypergeom"], method="fdr_bh")[1]

    enriched_peptides = results_df[results_df["adjusted_p_value"] < 0.05]
    return enriched_peptides


def calculate_function_score(oligopeptides, functional_counts):
    df = pd.DataFrame({
        "peptide": oligopeptides,
        "occurrences": [sum(functional_counts.get(peptide, [])) for peptide in oligopeptides]
    })

    df["length"] = df["peptide"].apply(len)

    scores = []
    for length, group in df.groupby("length"):
        group = group.sort_values("occurrences", ascending=False)
        group["rank_ratio"] = group["occurrences"].rank(pct=True, ascending=False)
        group["function_score"] = -np.log(group["rank_ratio"])
        scores.append(group)

    function_scores = pd.concat(scores)
    return function_scores


def plot_function_score_distribution(function_scores, indication):
    scores = function_scores["function_score"]
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 使用 stat="probability" 以显示相对频率
    sns.histplot(scores, bins=6, kde=True, color="#F6C8A8", alpha=0.7, stat="probability")
    
    plt.title(f"{indication}", fontsize=24)
    plt.xlabel("Function Score", fontsize=20)
    plt.ylabel("Relative Frequency (Percentage)", fontsize=20)
    
    # 将 y 轴刻度转换为百分比
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # 调整 x 和 y 轴刻度数字的字号
    plt.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度标签字号为 16
    plt.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度标签字号为 16
    
    plt.tight_layout()
    plt.show()


def plot_function_score_distribution(function_scores):
    scores = function_scores["function_score"]
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=30, kde=True, color="blue", alpha=0.7)
    plt.title("Frequency Distribution of Function Scores", fontsize=16)
    plt.xlabel("Function Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.show()


def main(indication):
    root_path = './pipeline/significance/'
    oligopeptide_file = f'{root_path}{indication}_peptides.txt'
    functional_idrs_file = f'{root_path}{indication}_positive_dataset.txt'
    non_functional_idrs_file = f'{root_path}{indication}_negative_dataset_1.txt'
    protein_universe_file = root_path+"protein_universe.txt"
    
    with open(oligopeptide_file, "r") as f:
        oligopeptides = [line.strip() for line in f]
    with open(functional_idrs_file, "r") as f:
        functional_idrs = [line.strip() for line in f]
    with open(non_functional_idrs_file, "r") as f:
        non_functional_idrs = [line.strip() for line in f]
    with open(protein_universe_file, "r") as f:
        protein_universe = [line.strip() for line in f]

    # occurence of oligopeptides
    functional_counts = count_oligopeptides_per_sequence(oligopeptides, functional_idrs)
    non_functional_counts = count_oligopeptides_per_sequence(oligopeptides, non_functional_idrs)
    universe_counts = count_oligopeptides_per_sequence(oligopeptides, protein_universe)

    # # total length
    total_functional_length = sum(len(seq) for seq in functional_idrs)
    total_universe_length = sum(len(seq) for seq in protein_universe)

    # # calculate Enrichment significance
    enriched_peptides = calculate_enrichment_significance(oligopeptides, functional_counts, non_functional_counts, universe_counts, total_functional_length, total_universe_length)
    print("Enrichment significance results:")
    print(enriched_peptides)

    # calculate Function score
    function_scores = calculate_function_score(enriched_peptides['peptide'], functional_counts)
    print("Function score results:")
    print(function_scores)

    # save results
    enriched_peptides.to_csv(f"{root_path}{indication}_enriched_peptides.csv", index=False)
    function_scores.to_csv(f"{root_path}{indication}_function_scores.csv", index=False)
    
    # plot
    plot_function_score_distribution(function_scores, indication)

if __name__ == "__main__":
    indications = ["Osteogenesis", "Angiogenesis", "Lipid metabolism", "Glucose metabolism", "Antiangiogenesis"]
    for indication in indications:
        main(indication)


# root_path = './pipeline/significance/'

# for indication in indications:
#     function_scores = pd.read_csv(f'{root_path}{indication}_function_scores.csv')
#     plot_function_score_distribution(function_scores, indication)


