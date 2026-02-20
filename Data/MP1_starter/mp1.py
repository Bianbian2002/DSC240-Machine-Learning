# Starter code for Winter 2025 DSC 240 MP1

def run_train_test(training_input, testing_input):
    """
    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """


    def _safe_div(num, denom):
        return num / denom if denom != 0 else 0.0

    def _centroid(rows, dim):
        sums = [0.0] * dim
        count = len(rows)
        if count == 0:
            return sums
        for row in rows:
            for i in range(dim):
                sums[i] += row[i]
        return [s / count for s in sums]

    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def _discriminant(ci, cj, x, tie_winner):
        # Hyperplane: (cj - ci) · x = (cj - ci) · midpoint
        n = [cj[k] - ci[k] for k in range(len(ci))]
        mid = [(ci[k] + cj[k]) * 0.5 for k in range(len(ci))]
        diff = _dot(n, x) - _dot(n, mid)
        if abs(diff) <= 1e-12:
            return tie_winner
        return 1 if diff > 0 else 0  # 1 => cj, 0 => ci

    # Parse training data
    dim, n_a, n_b, n_c = training_input[0]
    train_rows = training_input[1:]
    a_rows = train_rows[0:n_a]
    b_rows = train_rows[n_a:n_a + n_b]
    c_rows = train_rows[n_a + n_b:n_a + n_b + n_c]

    # Centroids for A, B, C
    c_a = _centroid(a_rows, dim)
    c_b = _centroid(b_rows, dim)
    c_c = _centroid(c_rows, dim)

    # Parse testing data
    _, t_a, t_b, t_c = testing_input[0]
    test_rows = testing_input[1:]
    true_labels = (["A"] * t_a) + (["B"] * t_b) + (["C"] * t_c)

    # Confusion counts per class
    counts = {
        "A": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "B": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "C": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
    }

    for x, true_label in zip(test_rows, true_labels):
        # First decide A vs B
        ab = _discriminant(c_a, c_b, x, tie_winner=0)  # 0 => A, 1 => B
        if ab == 0:
            # Decide A vs C
            ac = _discriminant(c_a, c_c, x, tie_winner=0)  # 0 => A, 1 => C
            pred = "A" if ac == 0 else "C"
        else:
            # Decide B vs C
            bc = _discriminant(c_b, c_c, x, tie_winner=0)  # 0 => B, 1 => C
            pred = "B" if bc == 0 else "C"

        for cls in ("A", "B", "C"):
            if pred == cls and true_label == cls:
                counts[cls]["tp"] += 1
            elif pred == cls and true_label != cls:
                counts[cls]["fp"] += 1
            elif pred != cls and true_label == cls:
                counts[cls]["fn"] += 1
            else:
                counts[cls]["tn"] += 1

    # Compute metrics per class, then average
    tpr = 0.0
    fpr = 0.0
    error_rate = 0.0
    accuracy = 0.0
    precision = 0.0
    for cls in ("A", "B", "C"):
        tp = counts[cls]["tp"]
        tn = counts[cls]["tn"]
        fp = counts[cls]["fp"]
        fn = counts[cls]["fn"]
        total = tp + tn + fp + fn
        tpr += _safe_div(tp, tp + fn)
        fpr += _safe_div(fp, fp + tn)
        error_rate += _safe_div(fp + fn, total)
        accuracy += _safe_div(tp + tn, total)
        precision += _safe_div(tp, tp + fp)

    tpr /= 3.0
    fpr /= 3.0
    error_rate /= 3.0
    accuracy /= 3.0
    precision /= 3.0

    return {
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision,
    }
