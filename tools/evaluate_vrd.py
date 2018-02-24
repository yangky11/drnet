import os
import argparse


def execute(cmd):
  print(cmd)
  os.system(cmd)


parser = argparse.ArgumentParser(description='')
parser.add_argument('model', type=str)
args = parser.parse_args()

print('testing..')
execute('python tools/test_predicate_recognition.py --def prototxts/test_%s.prototxt --net snapshots/%s.caffemodel' % (args.model, args.model))
print('evaluating recall@50')
execute('python tools/eval_triplet_recall.py --num_dets 50')
print('evaluating recall@100')
execute('python tools/eval_triplet_recall.py --num_dets 100')

