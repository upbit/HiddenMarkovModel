#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HiddenMarkovModel
"""

HMM_FILE_VERSION = "0.1"

class BaseHiddenMarkovModel(object):
	def __init__(self):
		"""
		states - a list/tuple of states
		observations - a list/tuple of the names of observable values
		starts - the probabilities of a starting state
					starts[state] = prob
		transitions - the probabilities to go from one state to another
					transitions[from_state][to_state] = prob
		emissions - the probabilities of an observation for a given state
					emissions[state][observation] = prob
		"""

	# functions to get stuff one-indexed
	get_state = lambda self, idx: [name for name, value in self.states.iteritems() if (value == idx)][0]
	get_state_idx = lambda self, str_state: self.states[str_state]
	get_observe = lambda self, idx: [name for name, value in self.observations.iteritems() if (value == idx)][0]
	get_observe_idx = lambda self, str_observe: self.observations[str_observe]

	# helper functions
	state_nums = lambda self: range(len(self.states))
	observe_nums = lambda self: range(len(self.observations))

	path2states = lambda self, path: [self.get_state(idx) for idx in path]
	index2observes = lambda self, observe_list: [self.get_observe(idx) for idx in observe_list]


	def transition(self, from_state, to_state):
		""" 转移矩阵 """
		return self.transitions[from_state][to_state]

	def emission(self, state, observed_idx):
		""" 状态/活动混淆矩阵 """
		return self.emissions[state][observed_idx]

	def _init_trellis(self, observe_list, forward=True):
		"""
		Init start probabilities of trellis, return trellis[step_of_observe][state]
		"""
		# 初始化一个 trellis[i:天数][j:状态数] 的空矩阵，用于存放结果
		trellis = [ [None for j in self.state_nums()] for i in range(len(observe_list)) ]

		if forward:
			# 向前初始概率，为(某状态初始概率 * 该状态下进行活动[0]的概率)
			init_func = lambda state: self.starts[state] * self.emission(state, self.get_observe_idx(observe_list[0]))
			# 对第一天的各种状态计算概率值
			for state in range(len(self.states)):
				trellis[0][state] = init_func(state)
		else:
			# 向后初始概率，这里默认最后状态直接到结束，所以转移概率为1.0
			init_func = lambda state: 1.0
			# 初始化末尾状态的概率值
			for state in range(len(self.states)):
				trellis[-1][state] = init_func(state)

		return trellis

	def forward(self, observe_list, return_trellis=False):
		"""
		Returns the probability of seeing the given `observations` sequence,
		using the Forward algorithm.
		"""
		trellis = self._init_trellis(observe_list, forward=True)

		# [0]天的概率已经在init中初始化，从[1]到[N]循环
		for step in range(1, len(observe_list)):
			for state in self.state_nums():
				# 某个状态的概率，是求前一天几个状态下，转移到当前状态，并且当前状态执行指定活动的概率之和
				# 因为当前状态进行[N]天的活动已知，所以可以先算出emission_prob
				emission_prob = self.emission(state, self.get_observe_idx(observe_list[step]))
				# 将前一天所有状态到当前状态的概率求和，其中old_state是前一天的状态
				trellis[step][state] = sum(
					trellis[step-1][old_state] * self.transition(old_state, state) * emission_prob \
						for old_state in self.state_nums()
				)

		# 向前概率为最后一天几种状态的概率之和
		final = sum(trellis[-1][state] for state in self.state_nums())
		return final if not return_trellis else (final, trellis)

	def backward(self, observe_list, return_trellis=False):
		"""
		Returns the probability of seeing the given `observations` sequence,
		using the Backward algorithm.
		"""
		trellis = self._init_trellis(observe_list, forward=False)

		# 向后算法是从末尾[N-1]向[0]状态推算，同样[N]的状态已经在init中初始化
		for step in reversed(range(0, len(observe_list)-1)):
			for state in self.state_nums():
				# 当前状态概率，是所有对后一天状态下，从当前状态转移到后一天的状态，
				# 并且后一天在指定状态下进行已知活动的概率求和。注意这里混淆矩阵是算的后一天的活动概率，不是当前的
				trellis[step][state] = sum(
					trellis[step+1][next_state] * self.transition(state, next_state) \
					* self.emission(next_state, self.get_observe_idx(observe_list[step+1])) \
						for next_state in self.state_nums()
				)

		# 向后概率为第一天在多种状态下进行已知活动，乘以该状态发生的概率之和
		final = sum(self.starts[state] * self.emission(state, self.get_observe_idx(observe_list[0])) \
					* trellis[0][state] for state in self.state_nums())
		return final if not return_trellis else (final, trellis)

	def viterbi(self, observe_list, return_trellis=False):
		"""
		Returns the most likely sequence of hidden states, for a given
		sequence of observations. Uses the Viterbi algorithm.

		# <Usage>
		# final, path, viterbis = hmm.viterbi(test_list, True)
		# hmm.print_trellis(viterbis)
		# print "%s: %5.2f%%, %s" % (test_list, final*100, [hmm.get_state(state).encode("ascii") for state in path])
		"""
		trellis = self._init_trellis(observe_list, forward=True)
		path = {state: [state] for state in self.state_nums()}

		for step in range(1, len(observe_list)):
			newpath = {}
			for state in self.state_nums():
				emission_prob = self.emission(state, self.get_observe_idx(observe_list[step]))
				(max_prob, max_state) = max([(trellis[step-1][old_state]
						* self.transition(old_state, state) * emission_prob,
						old_state) for old_state in self.state_nums()
				])
				trellis[step][state] = max_prob
				newpath[state] = path[max_state] + [state]
			path = newpath

		(final, state) = max([(trellis[-1][state], state) for state in self.state_nums()])
		return (final, path[state]) if not return_trellis else (final, path[state], trellis)

	def train_on_obs(self, json_config, observe_list, return_probs=False):
		"""
		Trains the model once, using the forward-backward algorithm. This
		function generate a new HMM configure and dump to `json_config` file.
		"""
		prob_forward, forwards = self.forward(observe_list, True)
		prob_backward, backwards = self.backward(observe_list, True)

		# posat[step][state] 定义在step时刻位于隐藏状态state的概率
		# 这里之所以可以直接除以prob_forward，是因为B-W算法已经求的了最大的概率，其实重复计算也是一样的
		prob_of_state_at_time = posat = [ [None for j in self.state_nums()] for i in range(len(observe_list)) ]
		for step in range(len(observe_list)):
			for state in self.state_nums():
				posat[step][state] = (forwards[step][state] * backwards[step][state]) / prob_forward

		# pot[step][state][next_state] 定义在step时刻位于隐藏状态state且下一时刻为next_state的概率
		# 注意没有最后一天的联合概率(没有step+1的状态)
		prob_of_transition = pot = [ [ [None for k in self.state_nums()] for j in self.state_nums() ] for i in range(len(observe_list)-1) ]
		for step in range(len(observe_list)-1):
			for state in self.state_nums():
				for next_state in self.state_nums():
					pot[step][state][next_state] = forwards[step][state] * self.transition(state, next_state) \
						* self.emission(next_state, self.get_observe_idx(observe_list[step+1])) * backwards[step+1][next_state]
					pot[step][state][next_state] /= prob_forward

		# 根据 posat/pot 矩阵重新估算HMM的参数
		for state in self.state_nums():
			self.starts[state] = posat[0][state]

		for state in self.state_nums():
			for next_state in self.state_nums():
				self.transitions[state][next_state] = sum(pot[step][state][next_state] for step in range(len(observe_list)-1)) \
					/ sum(posat[step][state] for step in range(len(observe_list)-1))

		# 混淆矩阵概率为在指定状态下，观测结果中是指定状态的概率，除以所有观测概率之和
		for state in self.state_nums():
			frac = sum(posat[step][state] for step in range(len(observe_list)))
			for ob_index in self.observe_nums():
				dem = sum(posat[step][state] for step in range(len(observe_list)) \
							if (self.get_observe_idx(observe_list[step]) == ob_index))
				self.emissions[state][ob_index] = dem / frac

		return (posat, pot) if not return_probs \
			else (prob_forward, prob_backward, posat, pot)


class JsonHiddenMarkovModel(BaseHiddenMarkovModel):
	def __init__(self, json_config):
		# init states/observations/... by json configs
		self.load_configure(json_config)

	# functions to load/save configure file
	def load_configure(self, json_config):
		from json import JSONDecoder
		configs = JSONDecoder().decode(open(json_config,'r').read())
		# self.states: 状态集合
		# self.observations: 活动集合
		# self.starts: 初始时为某状态的概率
		# self.transitions: 转移矩阵, 从某状态转移到另一状态的转移矩阵
		# self.emissions: 混淆矩阵, 某状态下做指定活动的概率
		self.states, self.observations, self.starts, self.transitions, self.emissions \
			= (configs["states"], configs["observations"], configs["starts"], configs["transitions"], configs["emissions"])

	def _save_configure(self, json_config, new_states, new_observations, new_starts, new_transitions, new_emissions, json_indent=2):
		from json import JSONEncoder
		if (json_indent > 0):
			json_string = JSONEncoder(indent=json_indent, sort_keys=True).encode({
				"__version__": HMM_FILE_VERSION,
				"states": new_states,
				"observations": new_observations,
				"starts": new_starts,
				"transitions": new_transitions,
				"emissions": new_emissions,
			})
		else:
			json_string = JSONEncoder(sort_keys=True).encode({
				"__version__": HMM_FILE_VERSION,
				"states": new_states,
				"observations": new_observations,
				"starts": new_starts,
				"transitions": new_transitions,
				"emissions": new_emissions,
			})

		open(json_config,'w').write(json_string)
		return json_string

	def dump_configure(self, json_config, json_indent=2):
		self._save_configure(json_config, self.states, self.observations, self.starts, self.transitions, self.emissions, json_indent)
