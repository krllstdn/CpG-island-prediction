### CpG Island prediction with Markov Models
CpG islands are DNA regions with a high frequency of CG dinucleotides, commonly found near the start of genes, particularly in promoter regions. Their presence can indicate transcription start sites and play a key role in gene regulation and epigenetic control. This project applies simple Markov models to detect CpG islands by modeling sequence transitions

## Key Components:

**Markov Model Training**: Trains separate first-order Markov models for CpG and null sequences
**Bayes Estimates**: Uses pseudocount of 1 for all transitions as specified
**Naive Bayes Classification**: Compares P(CpG|x) vs P(null|x) using Bayes' theorem
**Log-space Calculations**: Prevents numerical underflow by working in log space


## Algorithm Details:
**Initial Probabilities**: P(a) = (count(a) + 1) / (total_count + 5)
**Transition Probabilities**: P(b|a) = (count(ab) + 1) / (count(a·) + 5)
**Sequence Probability**: P(x|class) = P(x₁) × P(x₂|x₁) × ... × P(end|xₗ)
**Classification**: Predict CpG if P(CpG) × P(x|CpG) > P(null) × P(x|null)



