####################################
# jitter layer for data augmentation
'''
[example]:
layer {
  type: "Python"
  name: "ratemap"
  bottom: "ratemap"
  top: "ratemap"
  python_param {
    module: "nideep.layers.jitterlayer"
    layer: "JitterLayer"
    param_str:"{\'min_shift_f\':-8,\'max_shift_f\':8,\'min_shift_t\':-40,\'max_shift_t\':40}"
  }
        include{
         phase: TRAIN
        }
}
specify the shift parameters, min_ and max_ means the lower and upper 
bound of shift steps. 
positive values means shifting to right side.
actual shifting steps are generated randomly according to this shifting range.
'''
####################################

import caffe
import numpy as np
import random
from scipy.ndimage.interpolation import shift

class JitterLayer(caffe.Layer):
	def setup(self, bottom, top):
		'''
		bottom blob:
		[ratemap]:	Nx1xFxT
		[amsFeature]:	NxMxFxT
		1.check bottom/top blob vector size
		2.read in parameters
	    	shift in T,F(time,frequency) domain
		'''
		# check input
		if len(bottom) != 1:
			raise Exception("Need one input to apply jitter.")
		self.shift_f = 0 # shift over frequency domain
		self.shift_t = 0 # shift over time domain

#		self.forwardIter = 0

		params = eval(self.param_str)	# read in as dictionary
       		# set parameters according to the train_val.prototxt,
        	# use python built-in function for dictionary
    		self.min_shift_f = params.get('min_shift_f',0)
    		self.max_shift_f = params.get('max_shift_f',0)
		self.min_shift_t = params.get('min_shift_t',0)
		self.max_shift_t = params.get('max_shift_t',0)
	
	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape) # keep data shape from bottom blob

    	def forward(self, bottom, top):
 #   		self.forwardIter+=1
#		print 'iteration',self.forwardIter

		mu = 0.0
		var = 0.0
		flag = False
		# get dimension information
		batch_size=top[0].data.shape[0]
		modulation = top[0].data.shape[1]
		frequency = top[0].data.shape[2]
		time = top[0].data.shape[3]
		if ((not self.min_shift_f) and (not self.max_shift_f)):
			top[0].data[...]=bottom[0].data[...] #don't shift over frequency domain
		else:
			#generate noise and do the shift
			mu = np.mean(bottom[0].data[...])
			var = np.var(bottom[0].data[...])
#			print 'mu %f, var %d'%(mu,var)

			flag = True #wether these values are already computed

	        	#generate shift step randomly
	       		self.shift_f = random.randint(self.min_shift_f,self.max_shift_f)
	       		if self.shift_f==0:
	       			top[0].data[...]=bottom[0].data[...]
	       		else:
	        		noise_f = np.random.normal(mu,var,(batch_size,modulation,abs(self.shift_f),time))
#	        		print 'noise_f\n',noise_f
				top[0].data[...] = shift(bottom[0].data, [0,0,self.shift_f,0])
	        		if self.shift_f>0:
	        			top[0].data[:,:,0:self.shift_f,:] = noise_f
	        		else:
	        			top[0].data[:,:,self.shift_f:,:] = noise_f
#		print 'shift over frequency domain:'
#		print 'shift_f:',self.shift_f
#	    	print 'bottom\n',bottom[0].data[...]
#		print 'top:\n',top[0].data[...]

		if ((not self.min_shift_t) and (not self.max_shift_t)):
			top[0].data[...]=top[0].data[...] #don't shift over time domain
		else:
			if flag:
				#generate noise from mu and var(computed)
				#generate shift step randomly
		        	self.shift_t = random.randint(self.min_shift_t,self.max_shift_t)
		        	if self.shift_t==0:
		        		top[0].data[...]=top[0].data[...]
		        	else:
		        		noise_t = np.random.normal(mu,var,(batch_size,modulation,frequency,abs(self.shift_t)))
#		        		print 'noise_t\n',noise_t
					top[0].data[...] = shift(top[0].data, [0,0,0,self.shift_t])
		        		if self.shift_t>0:
		        			top[0].data[:,:,:,0:self.shift_t] = noise_t
		        		else:
		        			top[0].data[:,:,:,self.shift_t:] = noise_t
			else:				
				mu = np.mean(bottom[0].data[...])
	        		var = np.var(bottom[0].data[...])
#	        		print 'mu %f, var %d'%(mu,var)

	        		self.shift_t = random.randint(self.min_shift_t,self.max_shift_t)
		        	if self.shift_t==0:
		        		top[0].data[...]=top[0].data[...]
		        	else:
		        		noise_t = np.random.normal(mu,var,(batch_size,modulation,frequency,abs(self.shift_t)))
#		        		print 'noise_t\n',noise_t
					top[0].data[...] = shift(top[0].data, [0,0,0,self.shift_t])
		        		if self.shift_t>0:
		        			top[0].data[:,:,:,0:self.shift_t] = noise_t
		        		else:
		        			top[0].data[:,:,:,self.shift_t:] = noise_t
#		print 'shift over time domain:'
#		print 'shift_t:',self.shift_t
#		print 'top:\n',top[0].data[...]

	def backward(self, top, propagate_down, bottom):
	        pass # no backward pass

	

