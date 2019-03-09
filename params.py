class Params:
	def __init__(self):
		
		self.state_width =4**3 
		self.state_len = 1 #height of state
		self.episode_max_length = 100 # max time for which a apisode last
		
		self.num_actions=64
		self.num_frames = 1
		self.lr_rate = 0.001          # learning rate
		self.rms_rho = 0.9            # for rms prop
		self.rms_eps = 1e-9           # for rms prop
		self.num_epochs= 2000
		self.num_ex=160 # number of training examples
		self.num_test_ex =50 #number of test examples
		self.num_seq_per_batch= 50 #number of trajectories
		self.discount= 1
		self.batch_size = 4#nunber examples to run in parallel
	
		
		self.debug = False
		self.render = False
		self.pg_resume = None #path from where weights to be loaded
		self.render = False
