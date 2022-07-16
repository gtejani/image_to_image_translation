# from .inception import InceptionV3
from .metric import *

# create a function here for fid calculation and call that
def fid_scoreA(paths, batch_size=1, cuda=True, dims=2048):
	
	fid_scoreA = calculate_fid_given_paths(['./Real_A','./Fake_A'], batch_size=1, cuda=True, dims=2048)

	return fid_scoreA
    
    

def fid_scoreB(paths, batch_size=1, cuda=True, dims=2048):

	fid_scoreB = calculate_fid_given_paths(['./Real_B','./Fake_B'], batch_size=1, cuda=True, dims=2048)

	return fid_scoreB