import os
from random import choice
import sys

learning_rate_range = [2e-5, 1e-4, 5e-3]
weight_decay_range = [0., 1e-6]
momentum_range = [0., 0.5, 0.9]
batchsize_range = [16, 32]

params = set()
for i in range(int(sys.argv[1])):
  learning_rate = choice(learning_rate_range)
  weight_decay = choice(weight_decay_range)
  momentum = choice(momentum_range)
  batchsize = choice(batchsize_range)
  params.add((learning_rate, weight_decay, momentum, batchsize))

for param in params:
  (learning_rate, weight_decay, momentum, batchsize) = param
  exp_id = 'lr%f_l2%f_mom%f_bat%d' % (learning_rate, weight_decay, momentum, batchsize)
  template  = open('/home/yangky/sbatch_demo.sh').read()
  template = template.replace('JOBNAME', exp_id)
  template += "\npython -u main.py --learning_rate %f --weight_decay %f --momentum %f --batchsize %d --exp_id %s" % \
   (learning_rate, weight_decay, momentum, batchsize, exp_id)
  template += " 1>%s.out 2>%s.err" % (exp_id, exp_id)
  open('tmp_%s.sh' % exp_id, 'wt').write(template)
  #print(template)
  os.system('sbatch tmp_%s.sh' % exp_id)

