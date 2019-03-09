import time
import numpy as np
import theano
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import threading
import csv
#import simulation environment file
import policy_network
import sudoku
from multiprocessing import Process
from multiprocessing import Manager
import os,json

from environment import Environment


def init_accums(pg_learner):  # in rmsprop
	accums = []
	params = pg_learner.get_params()
	for param in params:
		accum = np.zeros(param.shape, dtype=param.dtype)
		accums.append(accum)
	return accums

def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):

	assert len(grads) == len(params)
	assert len(grads) == len(accums)
	for dim in range(len(grads)):
		accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
		params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))


def discount(x, gamma):
	"""
	Given vector x, computes a vector y such that
	y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
	"""
	out = np.zeros(len(x))
	out[-1] = x[-1]
	for i in reversed(range(len(x)-1)):
		out[i] = x[i] + gamma*out[i+1]
	assert x.ndim >= 1
	# More efficient version:
	# scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
	return out

def get_entropy(vec):
	entropy = - np.sum(vec * np.log(vec))
	if np.isnan(entropy):
		entropy = 0
	return entropy



def get_traj(agent, env, episode_max_length):
	"""
	Run agent-environment loop for one whole episode (trajectory)
	Return dictionary of results
	"""
	env.reset_grid()
	#print(sudoku.unflatten(ob))
	#call reset function of simulator
	obs = []
	acts = []
	rews = []
	final_obs=[]
	final_acts=[]
	final_rews=[]
	indices=[]
	#entropy = []
	info = []
	probs = []
	# for i in range(20):
	
		# env.seq_id += 1

	ob = env.current_grid
	#call the observe function from simulator
	#art_a = []
	finished_episode_len = 0
	for _ in range(episode_max_length):
		act_prob = agent.get_one_act_prob(ob)
		csprob_n = np.cumsum(act_prob)
		a = (csprob_n > np.random.rand()).argmax()

		obs.append(ob)
		#print('State>>',sudoku.unflatten(ob))
		#print('Action>>',a)  # store the ob at current decision making step
		acts.append(a)
		probs.append(act_prob)
		# plt1 = visualize_state(ob)
		# print('State at %d : ' % env.cur_time)
		# np.set_printoptions(linewidth=40*5, precision = 2, threshold=np.nan)
		# print(ob)
		# print(a+1)
		ob, rews, mistake, done= env.act(a)
		final_rews.append(rews)	
		# print(rews)
		#call the step function from simulator

	#    entropy.append(get_entropy(act_prob))
		
		if done: break

		
		
	#print('Training Reward>>',final_rews)	
		


	return {'reward': np.array(final_rews),
			'ob': np.array(obs),
			'action': np.array(acts),
			'prob' : probs
			}



def concatenate_all_ob(trajs, pa):

	timesteps_total = 0
	for i in range(len(trajs)):
		timesteps_total += len(trajs[i]['reward'])

	all_ob = np.zeros(
		(timesteps_total, 1, pa.state_len, pa.state_width),
		dtype=theano.config.floatX)

	timesteps = 0
	for i in range(len(trajs)):
		for j in range(len(trajs[i]['reward'])):
			all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
			timesteps += 1

	return all_ob



def concatenate_all_ob_across_examples(all_ob, pa):
	num_ex = len(all_ob)
	total_samp = 0
	for i in range(num_ex):
		total_samp += all_ob[i].shape[0]

	all_ob_contact = np.zeros(
		(total_samp, 1, pa.state_len, pa.state_width),
		dtype=theano.config.floatX)

	total_samp = 0

	for i in range(num_ex):
		prev_samp = total_samp
		total_samp += all_ob[i].shape[0]
		all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

	return all_ob_contact

def get_traj_worker(pg_learner, env, pa, result):
	trajs = []
	for i in range(pa.num_seq_per_batch):
		traj = get_traj(pg_learner, env, pa.episode_max_length)
		trajs.append(traj)
		# print(traj['action'])
		# print(traj['reward'], sum(traj['reward']))

	all_ob = concatenate_all_ob(trajs, pa)

	# Compute discounted sums of rewards
	rets = [discount(traj["reward"], pa.discount) for traj in trajs]
	maxlen = max(len(ret) for ret in rets)
	padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

	# Compute time-dependent baseline
	baseline = np.mean(padded_rets, axis=0)

	# Compute advantage function
	advs = [ret - baseline[:len(ret)] for ret in rets]
	all_action = np.concatenate([traj["action"] for traj in trajs])
	all_adv = np.concatenate(advs)

	all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
	all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

	result.append({"all_ob": all_ob,
				   "all_action": all_action,
				   "all_adv": all_adv,
				   "all_eprews": all_eprews,
				   "all_eplens": all_eplens})

def launch(pa, pg_resume=None, save_freq = 50, render=False, repre='image', end='no_new_job', test_only=False):

	
	

	pg_learners = []
	envs = []

	for ex in range(pa.num_ex):
		print("-prepare for env-", ex)
		env = Environment(ex)
		
		envs.append(env)

	for ex in range(pa.batch_size + 1):  # last worker for updating the parameters
		print("-prepare for worker-", ex)
		pg_learner = policy_network.PGLearner(pa)

		if pg_resume is not None:
			net_handle = open(pg_resume, 'rb')
			net_params = pickle.load(net_handle)
			pg_learner.set_net_params(net_params)

		pg_learners.append(pg_learner)

	accums = init_accums(pg_learners[pa.batch_size])

	print ('Preparing for Training from Scratch...')

	#   ref_discount_rews=slow_down_cdf.launch(pa,pg_resume=None,render=False,repre=repre,end=end)
	all_test_rews = []
	timer_start=time.time()


#	logs = open('/tmp/logs', 'a')
#	loglines = ''
	for iteration in range(380, pa.num_epochs+380):
		ps = []  # threads
		manager = Manager()  # managing return results
		manager_result = manager.list([])

		ex_indices = list(range(pa.num_ex))
	#	np.random.shuffle(ex_indices)

		all_ob=[]
		all_action=[]
		grads_all = []
		eprews = []
		eplens = []
		all_adv=[]
		all_eprews=[]
		all_eplens=[]

		ex_counter = 0
		for ex in range(pa.num_ex):
			ex_idx = ex_indices[ex]
			p = Process(target=get_traj_worker,
						args=(pg_learners[ex_counter], envs[ex_idx], pa, manager_result, ))
			ps.append(p)

			ex_counter += 1

			if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

				print(ex+1, "out of", pa.num_ex)

				ex_counter = 0

				for p in ps:
					p.start()

				for p in ps:
					p.join()

				result = []  # convert list from shared memory
				for r in manager_result:
					result.append(r)

				ps = []
				manager_result = manager.list([])
			
				all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
				all_action = np.concatenate([r["all_action"] for r in result])
				all_adv = np.concatenate([r["all_adv"] for r in result])
				grads = pg_learners[pa.batch_size].get_grad(all_ob, all_action, all_adv)

				grads_all.append(grads)

		
				all_eprews.extend([r["all_eprews"] for r in result])

				eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
				eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

		# assemble gradients
		grads = grads_all[0]
		for i in range(1, len(grads_all)):
			for j in range(len(grads)):
				grads[j] += grads_all[i][j]

		# propagate network parameters to others
		params = pg_learners[pa.batch_size].get_params()

		rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

		for i in range(pa.batch_size + 1):
			pg_learners[i].set_net_params(params)

		timer_end=time.time()
		print ("-----------------")
		print ("Iteration: \t %i" % iteration)
		print ("NumTrajs: \t %i" % len(eprews))
		print ("NumTimesteps: \t %i" % np.sum(eplens))
		print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
		print ("-----------------")
		# time.sleep(5)
		pg_resume = '/home/dell/rl_sudoku/4x4sudoku6_7_8_new_saved_weights/%s.pkl_' % str(iteration)
		if iteration % 10 == 0:
			param_file = open(pg_resume, 'wb')
			pickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
			param_file.close()

		if iteration % 20 == 0:
			logline = test(iteration, pa, pg_resume, pg_learners[pa.batch_size])
		#	loglines += logline
		#	if iteration % 20 == 0:
		#		logs.write(loglines)
		#		logs.flush()
		#		os.fsync(logs.fileno())
		#		loglines = ''
	#logs.close()

def test(it, pa ,pg_resume, pg_learner=None, episode_max_length=200):

	if pg_learner is None:
		pg_learner=policy_network.PGLearner(pa)
		if pg_resume is not None:
			net_handle = open(pg_resume, 'rb')
			net_params = pickle.load(net_handle)
			pg_learner.set_net_params(net_params)

	accuracy=0.
	#logline = str(it) + '\n'
	for ex in range(pa.num_test_ex):
		env = Environment(ex+pa.num_ex)
		ob=env.current_grid
		print(sudoku.unflatten(ob))
		
		print('Testing : ')
		
		acts = []
		probs = []
		
		rews = []
		final_obs=[]
		final_acts=[]
		final_rews=[]
		indices=[]
		json_array = []
		utils = 0
		suffer = []
		for _ in range(pa.episode_max_length):
			act_prob = pg_learner.get_one_act_prob(ob)
			csprob_n = np.cumsum(act_prob)
			a = np.argmax(act_prob)

			#################json
			
			
			# prev_waiting_tasks = env.waiting_tasks
			#################

			# plt1 = visualize_state(ob, pa, '/tmp/trajs/'+str(_)+'.jpg')
			# if _ < sum([len(i) for i in workloads[0]]):
			# 	print('Agent action: ', a)
			# 	man_act = input('Manual Action    :   ')
			# 	if man_act:
			# 		a = int(man_act)
			ob, rews, mistake, done= env.act(a)
			acts.append(a)
			probs.append(act_prob)
			final_rews.append(rews)
			if done:
				break
			##############logs
		if sum(final_rews)==0:
			accuracy+=1	
		if it % 20 == 0:
			print('Test Actions: ',acts)
			#print(probs)
			print('Reward : ', sum(final_rews))
			print('Full Reward: ',final_rews)
	print('Accuracy:',accuracy/pa.num_test_ex)
			


		# with open('/home/rnehra/json_logs/'+str(ex)+'.json', 'w') as json_file:
		# 	json.dump(json_array, json_file)

	



def main():
	import params
	import sys
	pa=params.Params()
	pg_resume='/home/dell/rl_sudoku/4x4sudoku_4_5_6_7_8_saved_weights/380.pkl_'
	#pg_resume = None
	test_only = False
	if len(sys.argv) == 2:
		pg_resume=sys.argv[1] #give the path of weights file
		test_only=True
	render=False
	launch(pa,pg_resume,render=render,repre='image',end='all_done', test_only=test_only)
#	test2(pa)

if __name__ =='__main__':
	main()



