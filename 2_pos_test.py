#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import operator
from hmm import *

debug_printf = lambda msg: sys.stdout.write(msg)
sorted_dict = lambda x: [val[0] for val in sorted(x.iteritems(), key=operator.itemgetter(1))]

def main():
	hmm = JsonHMM("train_word.hmm")
	print sorted_dict(hmm.states)
	print "state_num:%d, observe_num:%d" % (len(hmm.states), len(hmm.observations))
	
	#input_word = "有 农民 则 忧虑 ， 一旦 土地 被 徵收 、 房子 拆 了 ， 家族 从此 也 就 散 了 。"
	input_word = "研究生 研究 生命 的 起源 其实 不是 研究 。"
	test_list = input_word.decode("utf-8").strip().split()

	final, path, trellis = hmm.viterbi(test_list, True)
	print ">> call viterbi(test_list) return rate: %5.2f%%" % (final*100)

	for idx, word in enumerate(test_list):
		sys.stdout.write( "%s/%s " % (word, hmm.get_state(path[idx])) )
	sys.stdout.write("\n\n")

	#print hmm.path2states(path)

if __name__ == '__main__':
	main()
