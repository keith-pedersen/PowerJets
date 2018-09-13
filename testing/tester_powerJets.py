import numpy
import math
import powerJets as pJets
import copy
import random
import scipy.optimize
from kdp import Vec4, Vec3, Rot3, Boost

class PowerJets(pJets._PowerJets):
	def __init__(self, iniFileName):
		super().__init__(iniFileName)
		
	@staticmethod
	def SortJets_byE(jetVec):
		'''Return the jetVec sorted from highest to lowest energy.
		
		   Does not alter jetVec.'''
		return sorted(jetVec, key=(lambda p4: p4.x0), reverse=True)
		
	@staticmethod
	def FirstIndex_LessThan(theList, value, key = lambda x: x):
		'''Find the first index i where key(theList[i]) < value.
		
		   If the entire list is >= value, the list's length is returned. 
		   This allows the index i to be used as subList = theList[0:i]'''
		   
		for i in range(len(theList)):
			if(key(theList[i]) < value):
				return i
				
		return len(theList) # Catch all, in case the entire list is >= value
		
	@staticmethod
	def Durham_k2(jetA, jetB):
		'''The Durham algorithm "distance".'''
		if(jetA.p().Mag2() == 0.):
			raise ValueError("Durham_k2: jetA is not moving, there is no angle")
		elif(jetB.p().Mag2() == 0.):
			raise ValueError("Durham_k2: jetB is not moving, there is no angle")
		
		angle_AB = jetA.p().InteriorAngle(jetB.p())
		return 4.*min(jetA.x0, jetB.x0)*math.sin(0.5*angle_AB)**2
		
	@staticmethod
	def ClosestPair(jetVec, dist):
		'''Given some distance function, find the closest pair.'''
		dist_min = float('inf')
		indices = list()
		
		if(len(jetVec) > 2):
			# find the smallest measure
			for j in range(len(jetVec)):
				for i in range(j): # i < j
					dist_ij = dist(jetVec[i], jetVec[j])
					
					if(dist_ij < dist_min):
						dist_min = dist_ij
						indices = [i, j]
		elif(len(jetVec) == 2):
			return [0, 1]
					
		return indices # if len(jetVec) <= 1, this will return and empty list
		
	@staticmethod
	def ExclusiveCluster(jetVec_in, nJets=3, dist=Durham_k2):
		'''Cluster the list of incoming jets until nJets remain.'''
				
		if(nJets < 1):
			raise ValueError("ExclusiveCluster: nJets must be positive!")
		elif(nJets == 1):
			return [sum(jetVec_in, Vec4()),]
		
		jetVec = copy.deepcopy(jetVec_in)	
		
		while(len(jetVec) > nJets):
			# We need at least two values in the return to unpack, 
			# which is guaranteed by requiring nJets >= 2
			i, j = PowerJets.ClosestPair(jetVec, PowerJets.Durham_k2)
						
			jetVec[i] += jetVec[j]
			del jetVec[j]
			
		return jetVec
	
	@staticmethod
	def Delta(a, b, c):
		'''The kinematic quantity which shows up in a two-body decay.'''
		return math.sqrt((a + b + c)*(a + b - c)*(a - b + c)*(a - b - c))/a**2
	
	@staticmethod
	def KinematicParams(jetB, jetC):
		'''Calculate the dimensionless kinematic parameters for the splitting a -> b + c.'''
		jetA = jetB + jetC
		
		mass_A = jetA.Mass()

		u_b = jetB.Mass() / mass_A
		u_c = jetC.Mass() / mass_A

		u_bc = u_b + u_c
		u_b_star = u_b / u_bc
		
		z = jetB.x0 / jetA.x0
		beta_A = jetA.p().Mag() / jetA.x0
		
		if(beta_A < 1e-8):
			return [u_bc, u_b_star]
		else:
			betaDelta = beta_A * PowerJets.Delta(1., u_b, u_c)
		
			z_star = (z - 0.5*(1. + (u_b - u_c)*(u_b + u_c) - betaDelta))/betaDelta
			return [u_bc, u_b_star, z_star]
			
	@staticmethod
	def ActiveAngle(pol_orig, pol, axis):
		'''Calculate the angle of active, right-handed rotation about axis 
		   which takes pol_orig to pol.'''
		
		# axis is supplied, but we can also calculate it explicitly, and the two should match
		axis_calc = pol_orig.Cross(pol).Normalize()
		assert(math.fabs(1.-math.fabs(axis_calc.Dot(axis))) < 1e-14)
		
		return  pol_orig.InteriorAngle(pol)*numpy.sign(axis_calc.Dot(axis))
		
	@staticmethod
	def SolveJetParameters(jetVec_in, f_min = 0.05, nProngs = 3, validate = False):
		'''Take all jets above f_min and compress then (via Exclusive Durham) 
		   into 3 (or 4) prongs whose parameters are returned.
		
		   We use the convention that root daughter b is the first one to split.'''
		
		if((nProngs < 3) or (nProngs > 4)):
			raise ValueError("SolveJetParameters: Only 3 or 4 prongs supported")
		
		jetVec = PowerJets.SortJets_byE(copy.deepcopy(jetVec_in))
		
		##################################################################
		totalE = sum(jetVec, Vec4()).x0
		
		# Take all jets with energy fraction f about f_min.
		# If only two jets fit the bill, take the third so we can define a splitting plane.
		jetEnd = max(3, PowerJets.FirstIndex_LessThan(jetVec, 
			value = f_min * totalE, key = lambda p4: p4.x0))
			
		prongs = jetVec[0:jetEnd]
		
		print()
		for jet in prongs:
			print(jet.x0, jet.Eta(), jet.Phi())
			
		print()
		print(sum(prongs, Vec4()))
			
		if(len(prongs) < nProngs):
			nProngs = len(prongs)
		else:
			prongs = PowerJets.SortJets_byE(PowerJets.ExclusiveCluster(prongs, nJets = nProngs))
			
		print()
		for jet in prongs:
			print(jet.x0, jet.Eta(), jet.Phi())
			
		del jetEnd, totalE
		##################################################################
		CM = Boost(sum(prongs, Vec4()).BetaVec())
		
		# Boost the prongs into their CM frame
		for prong in prongs:
			CM.Backward(prong)
			
		print()
		for jet in prongs:
			print(jet.x0, jet.Eta(), jet.Phi())
		
		print()
		print(sum(prongs, Vec4()))
					
		del CM
		##################################################################
		prongs = PowerJets.SortJets_byE(prongs)
		
		if(validate):
			prongs_orig = copy.deepcopy(prongs)
			
		params_kinematic = list()
		polVec = list() # a list of splitting polarization (normalized)
		momVec = list() # a list of the mother's direction of travel (normalized)
		ijVec = list() # a list of the indices which clustered
				
		while(len(prongs) > 2):
			# Cluster the prongs with the smallest invariant mass
			i,j = PowerJets.ClosestPair(prongs, Vec4.Contract)
			params_kinematic.append(PowerJets.KinematicParams(prongs[i], prongs[j]))
			polVec.append(prongs[i].p().Cross(prongs[j].p()).Normalize())
			
			ijVec.append([i,j])
			prongs[i] += prongs[j]
			del(prongs[j])
			
			momVec.append(prongs[i].p().Copy().Normalize())
			
		# The last entry in momVec defines the two-jet axis (root daughter b)
		#    axis = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
		# The default orientation is along [0, 0, 1], with a polarization [1, 0, 0].
		# We want the rotation that takes [0, 0, 1] to this axis
		#  Crossing these two gets the rotation axis (which we then normalize)
		#     rot = [-sin(theta)sin(phi), sin(theta)cos(phi), 0] -> [-sin(phi), cos(phi), 0]
		theta = momVec[-1].Theta()
		phi = momVec[-1].Phi()
		
		rot = Rot3(Vec3(-math.sin(phi), math.cos(phi), 0), theta)
		
		# We rotate the default polarization, then calculate the polarization angle
		pol_default = rot(Vec3(1,0,0))
		omega = PowerJets.ActiveAngle(pol_default, polVec[-1], momVec[-1])
			
		params_orientation = [theta, phi, omega]
		del theta, phi, omega
		##################################################################
		addresses = [[False,],] # Root daughter b is the first to split (by convention)
				
		if(ijVec[-1][0] == 0):
			params_kinematic.append(PowerJets.KinematicParams(prongs[0], prongs[1]))
		else:
			params_kinematic.append(PowerJets.KinematicParams(prongs[1], prongs[0]))
			
		if(nProngs == 4):
			params_kinematic[0].append(PowerJets.ActiveAngle(polVec[-1], polVec[-2], momVec[-2]))
			
			# We have to figure out the address of the second splitting
			# The logic here is a bit convoluted, and this solution is certainly not elegant, 
			# but it seems to work
			if ijVec[0][0] in ijVec[-1]:
				if (ijVec[0][0] == ijVec[-1][0]):
					addresses.append([False,False])
				else:
					addresses.append([False,True])
			else:
				addresses.append([True,])
				
		params_kinematic = numpy.concatenate(params_kinematic[::-1]).tolist()
				
		# Check that the reconstructed system matches the original
		if(validate):
			jetDict = {"pileup_frac":False, "addresses":addresses}
			prongs_reco = PowerJets.SortJets_byE(PowerJets.GetJets(params_orientation + params_kinematic, 
				jetDict))
			totalE = sum(prongs_orig, Vec4()).x0
			
			for prong in prongs_orig:
				prong /= totalE
				
			error = math.fabs(sum(numpy.array(prongs_reco) - numpy.array(prongs_orig), Vec4()).Mass())
			assert(error < 1e-8)
				
		return params_kinematic, addresses, params_orientation
	
		
########################################################################
		
pythia = PowerJets("PowerJets.ini")

while(pythia.Next()):
	pythia.WriteAllVisibleAsTowers("towers.dat")
	
	tracks = sorted(pythia.Get_Tracks(), key = lambda p3: p3.Mag2(), reverse=True)
	towers = sorted(pythia.Get_Towers(), key = lambda p3: p3.Mag2(), reverse=True)
	
	print()
	print("tracks")
	for p3 in tracks:
		print(p3.Mag(), p3.Eta(), p3.Phi())
	
	print()
	print("towers")
	for p3 in towers[0:30]:
		print(p3.Mag(), p3.Eta(), p3.Phi())
	
	jets_raw = pythia.Cluster_FastJets(False)
	jets = pythia.Cluster_FastJets(True)
	
	totalE_raw = sum(jets_raw, Vec4()).x0
	totalE_sub = sum(jets, Vec4()).x0
	totalE = 0.
	
	for jet in jets:
		if(jet.x0 > .05 * totalE_sub):
			totalE += jet.x0	
	
	f_PU = (totalE_raw - totalE)/totalE_raw
	#~ f_PU = 1e-3
	#~ f_PU =(totalE_raw - 250.)/totalE_raw
	
	print(pythia.EventIndex())
	params_kinematic, addresses, params_orientation = PowerJets.SolveJetParameters(
		jets, f_min=0.05, nProngs=4, validate=True)
		
	jetParamDict = {"pileup_frac":True, "addresses":addresses, "hybrid":True}
		
	jetBounds = ((1e-3, 0.999),)*3 + ((-math.pi, math.pi),)
		
	xParams = [f_PU,] + params_orientation + params_kinematic
	xBounds = ((f_PU*0.9, f_PU*1.1),) + ((0., math.pi), (-math.pi, math.pi), (-math.pi, math.pi)) + jetBounds[0:2] + jetBounds[0:3]
		
	if(len(xBounds) < len(xParams)):
		xBounds += jetBounds
		
	assert(len(xBounds) == len(xParams))
	
	xParams_orig = xParams.copy()
			
	print("\nfitting")
	for lMax in [100]:
		print(lMax)
		fit = scipy.optimize.least_squares(
			pythia.Hl_Jet_error_vec,
			xParams,
			args = (jetParamDict, lMax),
			#~ method = 'trf',				
			bounds = tuple(zip(*xBounds)), # transpose
			jac = '3-point')
		xParams = fit.x
			#~ ftol = 1e-4)]
			#~ diff_step = 1e-10			
	
	
	lMax = 100
	
	Hl_Jet_orig = pythia.Hl_Jet(xParams_orig, jetParamDict, lMax)
	Hl_Jet = pythia.Hl_Jet(fit.x, jetParamDict, lMax)
	Hl_Obs = pythia.Hl_Obs(lMax)
	
	with open("test.dat", "w") as file:
		for lMinus1 in range(lMax):
			file.write("{} {:3e} {:3e} {:3e}\n".format(lMinus1 + 1, 
				Hl_Obs[lMinus1], Hl_Jet_orig[lMinus1], Hl_Jet[lMinus1]))	
	
	jets_seed = pythia.SortJets_byE(pythia.GetJets(xParams_orig, jetParamDict))			
	jets_reco = pythia.SortJets_byE(pythia.GetJets(fit.x, jetParamDict))	
	jetE = totalE_raw *(1. - fit.x[0])
	#~ jetE = totalE	

	print()
	print(fit.x[0], f_PU)
	print(len(fit.x))

	print()
	print("fit")
	for jet in pythia.SortJets_byE(PowerJets.ExclusiveCluster(jets_reco, 3, dist=Vec4.Contract)):
		print(jet.x0*jetE, jet.Eta(), jet.Phi())
	
	print()
	print("ME")
	for jet in pythia.SortJets_byE(pythia.Get_ME()):
		print(jet.x0, jet.Eta(), jet.Phi())
		
	#~ print()
	#~ for jet in jets:
		#~ print(jet, jet.Beta())
		
	print()
	print("anti-kt")
	for jet in jets[0:3]:
		print(jet.x0, jet.Eta(), jet.Phi())
	
	print()
	print("seed")
	for jet in jets_seed:
		print(jet.x0*(1.-f_PU)*totalE_raw, jet.Eta(), jet.Phi())
	
	continue
	
	
	#~ for n_pileup in 2.**numpy.arange(0, 10, 1):
	for n_pileup in [0., 16., 40., 100., 200.]:
		pythia.Set_PileupMu(n_pileup)
		for k_trial in range(5):
			pythia.Repeat()
			normalJets = pythia.Cluster_FastJets(False)
			print(normalJets[0])
			if(n_pileup):
				subtractedJets = pythia.Cluster_FastJets(True)
				print(subtractedJets[0])
				print(n_pileup, (numpy.sum(normalJets).x0 - numpy.sum(subtractedJets).x0)/(numpy.sum(normalJets).x0 - 250.), pythia.Hl_Obs(3))
		
	
