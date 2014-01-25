#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from hmm import *

debug_printf = lambda msg: sys.stdout.write(msg)

def append_word_to_dict(map_dict, count_dict, str_key):
	# 将key加入dict并累计位置和出现时间
	if not (str_key in map_dict):
		map_dict[str_key] = len(map_dict)
		count_dict[str_key] = 1
	else:
		count_dict[str_key] += 1

class HMM_TrainWordTags(JsonHMM):
	def __init__(self, train_data):
		""" 通过读取指定文件中的数据，训练生成HMM模型 """
		self.states = {}
		self.observations = {}

		state_counts = {}
		observations_counts = {}

		# 先遍历文件得到state/observe的数量(可以优化为字段表)
		for pos,word,tag in self.datafile(train_data):
			append_word_to_dict(self.states, state_counts, tag)
			append_word_to_dict(self.observations, observations_counts, word)

		# starts[] 某个词性出现在句首的概率
		self.starts = [ 0 for i in self.state_nums() ]
		# transitions[] 从一个词性转移到另一个词性的概率
		self.transitions = [ [ 0 for j in self.state_nums()] for i in self.state_nums() ]
		# emissions[] 已知词性时为指定单词的概率
		self.emissions = [ [ 0 for j in self.observe_nums()] for i in self.state_nums() ]

		# 接着分析输入文件，用统计方法生成hmm模型的参数
		first_loop = 1
		for pos,word,tag in self.datafile(train_data):
			state_index = self.get_state_idx(tag)
			observe_index = self.get_observe_idx(word)

			if (int(pos) == 1):		# 句首，记录该state在句首位置出现的次数
				self.starts[state_index] += 1
			self.emissions[state_index][observe_index] += 1

			if (first_loop == 1):
				first_loop = 0
				last_state = state_index
			else:
				self.transitions[last_state][state_index] += 1
				last_state = state_index

		self.start_count = sum(self.starts)
		assert self.start_count != 0
		self.starts = map(lambda x: float(x)/self.start_count, self.starts)

		for state in self.state_nums():
			state_count = state_counts[self.get_state(state)]
			self.transitions[state] = map(lambda x: float(x)/state_count, self.transitions[state])

		for state in self.state_nums():
			state_count = state_counts[self.get_state(state)]
			self.emissions[state] = map(lambda x: float(x)/state_count, self.emissions[state])

	def datafile(self, name, sep='\t', skip=0):
		"Read key,value pairs from file."
		for line in file(name):
			raw_text = line.strip()
			if (len(raw_text) <= 2):
				continue
			yield raw_text.split(sep)[skip:]

def main():
	#hmm = HMM_TrainWordTags("example_train.txt")
	hmm = HMM_TrainWordTags("train.txt")
	print "state_num:%d, observe_num:%d" % (len(hmm.states), len(hmm.observations))

	hmm.dump_configure("train_word.hmm", json_indent=0)
	print "save hmm to 'train_word.hmm'"

if __name__ == '__main__':
	main()
