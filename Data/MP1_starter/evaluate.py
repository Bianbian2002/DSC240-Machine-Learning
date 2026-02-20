#!/usr/bin/env python
import sys, os, os.path
import json
import signal
from contextlib import contextmanager
import fractions

# Some helper function to run tests
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def run_test(fn, train_input, test_input, expected, timeout, threshold=1e-10):
    acc = 0
    try:
        with time_limit(timeout):
            # TODO: Run test here
            predicted = fn(train_input, test_input)
            #print(predicted)
            acc = get_acc(expected, predicted)

    except TimeoutException as e:
        print("-- Failed. Taking too long.")

    return acc

def get_acc(expected, actual):
    num = 0
    for k in expected:
        if abs(actual[k] - expected[k]) <= .005:
            num += 1

    acc = float(num) / 5.0

    return acc


if __name__ == "__main__":

    # Load student code
    from mp1 import run_train_test

    reference_dir = './data/'
    output_dir = ''
    # Import test and solution files
    train_fnames = ['training1.txt']
    test_fnames = ['testing1.txt']

    train_fnames = [os.path.join(reference_dir, x) for x in train_fnames]
    test_fnames = [os.path.join(reference_dir, x) for x in test_fnames]

    results_path = os.path.join(output_dir, 'scores.txt')

    solutions = []
    with open(os.path.join(reference_dir, 'soln.json'), 'r') as f:
        solutions = json.load(f)

    total_acc = 0
    for idx, (train_fname, test_fname, soln) in enumerate(zip(train_fnames, test_fnames, solutions)):
        test_name = "test" + str(idx)
        train = []
        test = []

        with open(train_fname, "r") as f:
            train = [[float(y) for y in x.strip().split(" ")] for x in f]
            train[0] = [int(x) for x in train[0]]

        with open(test_fname, "r") as f:
            test = [[float(y) for y in x.strip().split(" ")] for x in f]
            test[0] = [int(x) for x in test[0]]

        #evaluate student code
        acc = run_test(run_train_test, train, test, soln, 5)


        total_acc += acc

    total_acc = total_acc / 1.0
    print(total_acc)


