import math
from collections import defaultdict, Counter

class CpGClassifier():
    def __init__(self):
        self.alphabet = {'A', 'C', 'G', 'T', 'end'}
        self.cpg_initial_probs = {}
        self.null_initial_probs = {}
        self.cpg_transition_probs = {}
        self.null_transition_probs = {}
        self.cpg_prior = 0.0
        self.null_prior = 0.0

    def train(self, cpg_sequences, null_sequences):
        """
        cpg_sequences
        null_sequences - sequences in the rest of the genome
        """
        total_seq_num = len(cpg_sequences) + len (null_sequences)
        self.cpg_prior = len(cpg_sequences)/total_seq_num
        self.null_prior = len(null_sequences)/total_seq_num

        self.cpg_initial_probs, self.cpg_transition_probs = self._train_mm(cpg_sequences)
        self.null_initial_probs, self.null_transition_probs = self._train_mm(null_sequences)


    def _train_mm (self, sequences):
        """Train 1st order Markov Model from the given sequences (one class)"""

        initial_counts = Counter()  # count fist symbols
        transition_counts = defaultdict(Counter)  # count transitions (including transitions to 'end')

        # TODO: Count all symbols for normalization
        for seq in sequences:
            if len(seq) == 0:
                continue

            initial_counts[seq[0]] += 1

            for i in range(len(seq) - 1):
                from_symbol = seq[i]
                to_symbol = seq[i + 1]
                transition_counts[from_symbol][to_symbol] += 1

            if len(seq) > 0:
                last_symbol = seq[-1]
                transition_counts[last_symbol]['end'] += 1
        
        # TODO: Calculate initial probabilities with pseudocount=1
        initial_probs = {}
        total_initial = sum(initial_counts.values()) + len(self.alphabet)  # +5 for pseudocounts
        
        for symbol in self.alphabet:
            if symbol == 'end':
                initial_probs[symbol] = 1.0 / total_initial  # pseudocount only
            else:
                initial_probs[symbol] = (initial_counts[symbol] + 1.0) / total_initial

        # TODO: Calculate transition probabilities with pseudocount = 1
        transition_probs = {}
        for from_symbol in self.alphabet:
            if from_symbol == 'end':
                continue  # No transitions from 'end'
                
            transition_probs[from_symbol] = {}
            
            # Total count for this from_symbol (including pseudocounts)
            total_from_symbol = sum(transition_counts[from_symbol].values()) + len(self.alphabet)
            
            for to_symbol in self.alphabet:
                count = transition_counts[from_symbol][to_symbol] + 1.0  # pseudocount
                transition_probs[from_symbol][to_symbol] = count / total_from_symbol

        return initial_probs, transition_probs

    def _calculate_sequence_probability(self, sequence, initial_probs, transition_probs):
        """Calculate the probability of a sequence given the model."""
        
        if len(sequence) == 0:
            return 0.0  # log probability

        # initial probability (log space)
        log_prob = math.log(initial_probs[sequence[0]])
        
        # add transition probabilities
        for i in range(len(sequence) - 1):
            from_symbol = sequence[i]
            to_symbol = sequence[i + 1]
            log_prob += math.log(transition_probs[from_symbol][to_symbol])
        
        # add transition to end
        if len(sequence) > 0:
            last_symbol = sequence[-1]
            log_prob += math.log(transition_probs[last_symbol]['end'])
        
        return log_prob

    def classify(self, sequence):
        """Classify a sequence as CpG (1) or null (0)."""
        cpg_log_prob = math.log(self.cpg_prior) + \
                       self._calculate_sequence_probability(sequence, self.cpg_initial_probs, self.cpg_transition_probs)
        
        null_log_prob = math.log(self.null_prior) + \
                        self._calculate_sequence_probability(sequence, self.null_initial_probs, self.null_transition_probs)
        
        # Return 1 if P(CpG|x) > P(null|x), otherwise 0
        return 1 if cpg_log_prob > null_log_prob else 0


def load(filename:str):
    sequences = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                sequence = line.strip().upper()
                if sequence:  # Skip empty lines
                    sequences.append(sequence)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []
    return sequences

def load_labels(filename):
    labels = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                label = int(line.strip())
                labels.append(label)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []
    return labels

def calculate_metrics(predictions, true_labels):
    """Calculate accuracy, precision, and recall."""
    
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    wrong = len(predictions) - correct
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
    
    # calculate precision and recall for CpG class (class 1)
    true_positives = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    false_positives = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    false_negatives = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return correct, wrong, accuracy, precision, recall

def main():
    print("loading training data...")
    cpg_train = load("data/cpg_train.txt")
    null_train = load("data/null_train.txt")

    if not cpg_train or not null_train:
        print("Error: Could not load training data.")
        return

    print(f"Loaded {len(cpg_train)} CpG sequences and {len(null_train)} null sequences")

    print("Training classifier...")
    classifier = CpGClassifier()
    classifier.train(cpg_train, null_train)

    print("loading test data...")
    labels = load_labels("data/classes_test.txt")
    test_seq = load("data/seqs_test.txt")

    if not test_seq or not labels:
        print("Error: Could not load test data.")
        return
    
    if len(test_seq) != len(labels):
        print("Error: Number of test sequences and labels don't match.")
        return
    print(f"Loaded {len(test_seq)} test sequences")

    print("Making predictions...")
    predictions = []
    for seq in test_seq:
        pred = classifier.classify(seq)
        predictions.append(pred)
    
    correct, wrong, accuracy, precision, recall = calculate_metrics(predictions, labels)
    
    # Save predictions
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    # Save accuracy metrics
    with open('accuracy.txt', 'w') as f:
        f.write(f"{correct}\n")
        f.write(f"{wrong}\n")
        f.write(f"{accuracy:.6f}\n")
        f.write(f"{precision:.6f}\n")
        f.write(f"{recall:.6f}\n")
    
    print("Files 'predictions.txt' and 'accuracy.txt' have been created.")
    
    print(f"Results:")
    print(f"Correct predictions: {correct}")
    print(f"Wrong predictions: {wrong}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    

if __name__ == "__main__":
    main()


