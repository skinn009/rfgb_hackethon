from GradientBoosting import GradientBoosting
import random

def main():
    '''main method that runs boosting'''

    bk = ["male(+name)","childof(+name,+name)","siblingof(+name,-name)","father(name,name)"]
    facts = []
    pos = []
    neg = []

    with open("Father/train/train_facts.txt") as f:
        facts = f.read().splitlines()
        facts = [item[:-1] for item in facts]

    with open("Father/train/train_pos.txt") as p:
        pos = p.read().splitlines()
        pos = [item[:-1] for item in pos]

    with open("Father/train/train_neg.txt") as n:
        neg = n.read().splitlines()
        neg = [item[:-1] for item in neg]

    ratio = len(neg)/float(len(pos))
    
    
    if ratio > 1:
        prob = 2*len(pos)/float(len(neg))
        neg = [item for item in neg if random.random() < prob]

    clf = GradientBoosting(treeDepth = 3, trees = 3)
    clf.setTargets(["father"])
    clf.learn_clf(facts,pos,neg,bk)

main()
