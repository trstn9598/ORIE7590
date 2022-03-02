import numpy as np

class BirthDeathBessel:

	def __init__(self, t = 0, x0 = [0]):
		x0 = np.array(x0)
		if x0.ndim == 0:
			x0 = np.array([x0])
		self.num = len(x0)
		self.t = t
		self.times = [[t] for n in range(self.num)]
		self.states = [[x0[n]] for n in range(self.num)]

	def next_step(self, sync = False):
		last_states = [states[-1] for states in self.states]
		if sync:
			expo = np.random.exponential()
			unif = np.random.rand() 
			return [[expo/(2*s + 1), s + (-1)**(int(unif < s/(2*s + 1)))] for s in last_states]
		else:
			return [[np.random.exponential()/(2*s + 1), s + (-1)**(int(np.random.rand() < s/(2*s + 1)))] for s in last_states]


	def simulate(self, dt = 0, sync = False):
		self.t = self.t + dt
		remaining_time = [dt for n in range(self.num)]
		while max(remaining_time) > 0:
			steps = self.next_step(sync = sync)
			for n in range(self.num):
				time, state = steps[n]
				if time > remaining_time[n]:
					remaining_time[n] = 0
				else:
					self.times[n].append(self.times[n][-1] + time)
					self.states[n].append(state)
					remaining_time[n] -= time

	def states_at_time(self, T = 0):
		T = max(min(self.t, T), self.times[0][0])
		size = len(self.states)
		indices = []
		for n in range(size):
			times = self.times[n]
			index = len(times) - 1
			for j in range(len(times)):
				if times[j] <= T:
					index = j
			indices.append(index)
		return [ self.states[n][indices[n]] for n in range(size)]