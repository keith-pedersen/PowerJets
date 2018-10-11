import numpy
import math
import itertools
import powerJets as pJets
import copy
import random
import scipy.optimize
from kdp import Vec4, Vec3, Rot3, Boost

# It was discovered that using a RelDiffThreshold actually prevented 
# accurate prong reconstruction with very boosted prongs. 
# We also determined (see test_Mass in libkdp) that this is actually 
# the most accurate result (provided one accounts for negative mass).
Vec4.SetLengthRelDiffThreshold(0.)

class PowerJets(pJets._PowerJets):
	def __init__(self, iniFileName : str):
		super().__init__(iniFileName) # Call the super constructor
		
	@staticmethod
	def SortByE(jetVec):
		'''Return jetVec sorted from highest to lowest energy.'''
		   
		return sorted(jetVec, key=(lambda p4: p4.x0), reverse=True)
		
	@staticmethod
	def EnsembleMass2(jetVec):
		'''Return a generator that calculates all "corrected" dijet masses.
		
		Mcorrected_ij**2 = (p_i**2 + p_j**2)/(n-1) + 2*p_i.p_j
		
		Summing Mcorrected_ij**2 for all i < j gives the total square mass of the system.
		Verifying the corrected mass is safer than verifying all p_i * p_j and p_i**2, 
		because individual jet mass p_i**2 may be vanishing (creating large relative error).'''
		
		nFactor = 1. / (len(jetVec) - 1)
		return (2.*jetVec[i].Contract(jetVec[j]) + (jetVec[i].Mass2() + jetVec[j].Mass2())*nFactor
			for i in range(len(jetVec)) for j in range(i))
		
	# Testing reveals that lambda expressions are safe as default arguments (immutable)
	@staticmethod
	def FirstIndex_LessThan(theList, value, key = lambda x: x):
		'''Find the first index i where key(theList[i]) < value.
		
		   If the entire list is >= value, the list's length is returned. 
		   This allows the index i to be used as subList = theList[0:i]'''
		   
		for i, element in enumerate(theList):
			if(key(element) < value):
				return i
		else: # if the loop exists without a break, the entire list is >= value
			return len(theList)
		
	@staticmethod
	def Durham_k2(jetA, jetB):
		'''The Durham algorithm "distance": 2*min(E_a, E_b)*(1-cos(theta_AB)).'''
		
		if(jetA.p().Mag2() == 0.):
			raise ValueError("Durham_k2: jetA is not moving, there is no angle")
		elif(jetB.p().Mag2() == 0.):
			raise ValueError("Durham_k2: jetB is not moving, there is no angle")
		
		angle_AB = jetA.p().InteriorAngle(jetB.p())
		return 4.*min(jetA.x0, jetB.x0)*math.sin(0.5*angle_AB)**2
		
	@staticmethod
	def ClosestPair(jetVec, dist):
		'''Given a list of jets and a distance function, 
		   find the closest pair (returned as an tuple of the two indices).'''
		
		# Find every unique jet pair via a generator that spans unique pairs of indices
		pairs = ((i, j) for j in range(len(jetVec)) for i in range(j))
		
		return min(pairs, key = lambda ij: dist(jetVec[ij[0]], jetVec[ij[1]]), 
			default=tuple()) # if len(jetVec) <= 1, return and empty list
		
	@staticmethod
	def ExclusiveCluster(jetVec_in, nJets=3, dist=Durham_k2):
		'''Cluster the list of incoming jets until nJets remain.'''
				
		if(nJets < 1):
			raise ValueError("ExclusiveCluster: nJets must be positive!")
		elif(nJets == 1): # cluster all 
			return list(sum(jetVec_in, Vec4()))
		
		# must deep copy since we are calling += directly on elements, 
		# which will alter the incoming list
		jetVec = copy.deepcopy(jetVec_in)
		
		while(len(jetVec) > nJets):
			# We need at least two values in the return to unpack, 
			# which is guaranteed by requiring nJets >= 2 above
			i, j = PowerJets.ClosestPair(jetVec, dist=PowerJets.Durham_k2)
						
			jetVec[i] += jetVec[j]
			del jetVec[j]
			
		return jetVec
	
	@staticmethod
	def Delta(a, b, c):
		'''The expression which shows up in a two-body kinematics.'''
		return math.sqrt((a + b + c)*(a + b - c)*(a - b + c)*(a - b - c))/a**2
		
	@staticmethod
	def SolveJetParameters(jetVec, f_min : float = 0.05, 
		nProngs : int = 3, validate : bool = False):
		'''Take from jetVec all jets above with energy fraction above f_min and 
		   compress then (via Exclusive Durham) into nProngs 
		   whose parameters are returned.
		
		   We use the conventions defined in ShowerParticle.'''
		
		## We begin with an inner class that uses a binary tree to
		## determine the jet parameters (following a Durham clustering).
		##################################################################
		##################################################################
				
		class _ClusterNode(Vec4):
			'''An extension of kdp.Vec4 to permit the automatic determination 
			of splitting parameters for a given ensemble of jets.
			
			We use ClusterNode to cluster prongs into a binary tree, 
			which is primarily useful for managing prong addresses.'''
			
			##################################################################
			
			# The Vec4 ctor takes the Vec4 component (4 x float).
			# We want to take a fully built Vec4 as this ctor's argument.
			# Per Cython documentation, this requires defining __new__, 
			# whose presence prevents the automatic call to super().__cinit__
			@staticmethod
			def __new__(cls, p4 : Vec4):
				# Pass the components to the Vec4 ctor
				return Vec4.__new__(cls, p4.x0, p4.x1, p4.x2, p4.x3)
				
			##################################################################
			
			def __init__(self, p4 : Vec4):
				# Three "pointers" to define a binary tree traversable up/down
				self.mother = None
				self.b = None
				self.c = None
				# The polarization of the splitting: b.Cross(c).Normalize()
				self.pol = None
				# The splitting parameters: [u_bc, u_b_star, z_star, phi]
				self.params = None
			
			##################################################################
			
			def IsLeaf(self):
				'''Is the node a leaf (not split, terminal)'''
				return (self.b is None)
				
			##################################################################
				
			def IsBranch(self):
				'''Is the node a branch (split)'''
				return (not self.IsLeaf())
				
			##################################################################
						
			# Can't use default arguments when the type is mutable
			def GetParameters(self, address : list, splitting : list, addresses : list):
				'''Append this node's parameters to the running lists.'''
				
				# If this node is a leaf, it has no parameters to append
				if(self.IsBranch()):
					# Append address and determine omega (now that we know all polarizations).
					# The root node has no address and no polarization, so we skip this scope
					if(len(address)):
						assert(self.mother is not None) # A not-root node should have a mother
						
						addresses.append(address)
						
						# Find the polarization rotation angle
						omega = self.ActiveAngle(pol_orig = self.pol, pol = self.b.pol, axis = self.p())
						assert(self.b.pol is self.c.pol)
						
						# The 1st splitting (root = 0th) appends omega to the root orientation, 
						# because it uses omega = 0 for its own parameters
						if(len(addresses) == 1):
							# The mother of the 1st splitting should be root
							assert(self.mother.mother is None)
							assert(hasattr(self.mother, "orientation"))
												
							# For orientation, omega is the rotation about root daughter b.
							# So if this is root daughter c, we have reverse the sign of omega
							self.mother.orientation.append(omega * (-1. if(address[-1]) else 1.))
							
							# root.orientation is used to rotate the polarization of both root daughters, 
							# so that they start with the polarization of the first splitting.
							# Thus, we set both root daughters to have this splitting's polarization.
							self.pol = self.b.pol
							# Also change the sister
							(self.mother.b if address[-1] else self.mother.c).pol = self.pol
							assert(self.mother.b.pol is self.mother.c.pol)
							
						else: # All other splittings add omega to their splitting parameters
							self.params.append(omega)
							
						del omega
					############################################################
									
					splitting += self.params
					
					# By convention, the address of self.b is (self.address + False)
					self.b.GetParameters(address + [False,], splitting, addresses)
					self.c.GetParameters(address + [True,], splitting, addresses)
					
					###############################################################
					
					# Only the root node has orientation, and only it needs to return
					if(hasattr(self, "orientation")):
						return splitting, addresses, self.orientation
			
			# We have to use string literals to annotate an incomplete type (such as this class)
			@staticmethod
			def Cluster(b : '_ClusterNode', c : '_ClusterNode', isRoot : bool = False):
				'''Cluster two ClusterNodes together, determining their kinematic parameters'''
				
				# Define a new ClusterNode from the 4-vector sum
				a = _ClusterNode(b + c) # calls Vec4.__plus__
						
				# Bind the two daughters to their new mother
				b.mother = a
				c.mother = a
				
				# Bind the mother to the two daughters
				a.b = b
				a.c = c						
				
				###############################################################
				
				# Sometimes a zero mass is slightly negative from rounding error;
				# other times it is slightly positive. Using max(0., a.Mass())
				# allows the latter situations to retain a vanishing mass, but not the former.
				# To treat all rounding error equal, we take fabs(mass).
				# Note that 
				mass_A = math.fabs(a.Mass())

				# Mass fractions
				u_b = math.fabs(b.Mass()) / mass_A
				u_c = math.fabs(c.Mass()) / mass_A
				assert((u_b >= 0.) and (u_b <= 1.))
				assert((u_c >= 0.) and (u_c <= 1.))
				
				del mass_A
				
				###############################################################
				
				# Daughter total mass fraction
				u_bc = u_b + u_c
				assert(u_bc <= 1.)
			
				# Relative mass fraction of daughter b; if both daughters are massless,
				# set u_b_star to zero by convention (any number will do, since u_bc = 0)
				u_b_star = u_b / u_bc if (u_bc > 0.) else 0.
				assert(u_b_star <= 1.)
			
				beta_A = min(1., a.Beta()) # The speed of the mother
							
				if(isRoot): # The root node behaves slightly differently
					
					# Sanity check: if this is the root node, then we should be in the CM/rest frame.
					# 3 cm/s is slow enough to leave room for rounding error
					assert(beta_A < 1e-10)
					
					a.params = [u_bc, u_b_star] # The root node does not have z*
					
					# The root node also records the orientation of the entire tree;
					# by convention, the dijet axis is parallel to root daughter b
					#     axis = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
					theta = b.Theta()
					phi = b.Phi()
					
					# Set orientation. This gives the root object a different 
					# attribute than every other ClusterNode. 
					# Python allows you to do this, so why not?
					a.orientation = [theta, phi]		
					
					# The ensemble defaults to the following orientation
					#    axis = [0, 0, 1],      pol = [1, 0, 0].
					# To orient, we use the rotation that takes [0, 0, 1] to 
					#    [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)].
					# Crossing these two vectors gives the rotation axis (which we then normalize)
					#    rot = [-sin(theta)sin(phi), sin(theta)cos(phi), 0] 
					#        = [-sin(phi), cos(phi), 0]
					# Using this rotation axis, we rotate by angle +theta
					b.pol = Vec3(1,0,0) # set to default, then rotate by theta
										
					Rot3(Vec3(-math.sin(phi), math.cos(phi), 0), theta)(b.pol)
					c.pol = b.pol					
					
					del theta, phi					
				else:
					assert(beta_A > 1e-10) # Only the root node should be at rest
					
					b.pol = b.p().Cross(c.p()).Normalize()
					c.pol = b.pol
					
					# energy fraction
					z = b.x0 / a.x0
					assert((z >= 0.) and (z <= 1.))
					
					# To convert to zStar, we refer to the math in KDP thesis; 
					# the following term shows up twice
					betaDelta = beta_A * PowerJets.Delta(1., u_b, u_c)
					
					z_star = (z - 0.5*(1. + (u_b - u_c)*(u_b + u_c) - betaDelta))/betaDelta
					assert((z_star <= 1.) and (z_star >= 0.))
					
					a.params = [u_bc, u_b_star, z_star]
					
					del z, z_star, betaDelta
					
				del u_b, u_c, u_bc, u_b_star, beta_A
					
				###############################################################
					
				return a
				
			@staticmethod
			def ActiveAngle(pol_orig : Vec3, pol : Vec3, axis : Vec3):
				'''Calculate the angle of active, right-handed rotation about axis 
					which takes pol_orig to pol.
					
					None of the vectors are required to be normalized'''
					
				# Axis is supplied, but we need to calculate it explicitly to
				# determine the sign of the rotation.
				axis_calc = pol_orig.Cross(pol)
				# Verify that the two axes are parallel/anti-parallel
				dotAxis = axis_calc.Dot(axis) / math.sqrt(axis_calc.Mag2() * axis.Mag2())
								
				if(math.fabs(1. - math.fabs(dotAxis)) > 1e-14):
					raise ValueError("PowerJets.ClusterNode.ActiveAngle: axis is not orthogonal to the two polarizations")
				
				# The interior angle is always positive; the sign of the RH rotation
				# is positive if axis_calc || axis, so use the sign of their dot product
				return pol_orig.InteriorAngle(pol)*numpy.sign(dotAxis)
		
		##################################################################
		# Done with _ClusterNode definition
		##################################################################
		
		##################################################################
		# Begin SolveJetParameters
		##################################################################
		# 1. Find the raw prongs and cluster them until nProngs remain
		
		if(nProngs < 3):
			raise ValueError("PowerJets.SolveJetParameters: nProngs should be >= 3, so we can define a splitting plane.")
		if(len(jetVec) < 3):
			raise ValueError("PowerJets.SolveJetParameters: len(jetVec) should be >= 3, so we can define a splitting plane.")
			
		jetVec = PowerJets.SortByE(jetVec)
		totalE = sum(jetVec, Vec4()).x0
		
		# Take all jets with energy fraction f about f_min.
		# If only two jets fit the bill, take the third so we can define a splitting plane.
		jetEnd = max(min(3, len(jetVec)), PowerJets.FirstIndex_LessThan(jetVec, 
			value = f_min * totalE, key = lambda p4: p4.x0))
			
		# Deep copy the 4-momenta so we can alter them without side effects
		prongs = copy.deepcopy(jetVec[0:jetEnd])
		
		# Cluster prongs until there are nProngs (or decrease nProngs if there aren't enough prongs)
		if(len(prongs) < nProngs):
			nProngs = len(prongs)
		else:
			prongs = PowerJets.SortByE(PowerJets.ExclusiveCluster(prongs, nJets = nProngs))
			
		if(validate): # Remember the original prongs if we are validating the result
			prongs_orig = PowerJets.SortByE(copy.deepcopy(prongs))
			
		del jetEnd, totalE
		##################################################################
		# 2. Boost the system into its CM frame
		
		beta_CM = sum(prongs, Vec4()).BetaVec()
		boost = Boost(beta_CM)
		
		boost_params = [beta_CM.Mag(), beta_CM.Eta(), beta_CM.Phi()]
		
		# Boost the prongs into their CM frame
		for prong in prongs:
			boost.Backward(prong)
			
		p4_CM = sum(prongs, Vec4())
		assert(p4_CM.p().Mag() < 1e-8 * p4_CM.x0)
					
		del beta_CM, boost, p4_CM
		##################################################################
		# 3. Convert the prongs to ClusterNode (implicit deep copy), 
		#    cluster them, then retrieve the parameters
		prongs = [_ClusterNode(p4) for p4 in PowerJets.SortByE(prongs)]
			
		while(len(prongs) > 2): # Keep going till there are two prongs
			# At each step, cluster the two closest prongs
			i,j = PowerJets.ClosestPair(prongs, dist = PowerJets.Durham_k2) # dist = Vec4.Contract)
			prongs[i] = _ClusterNode.Cluster(prongs[i], prongs[j])
			del prongs[j]
		
		# Cluster the root node	
		prongs[0] = _ClusterNode.Cluster(prongs[0], prongs[1], isRoot=True)
		del prongs[1]
		
		# Seed GetParameters with emtpy lists (can't use default parameters for mutable type)
		splitting, addresses, orientation = prongs[0].GetParameters(list(), list(), list())
				
		##################################################################
		# 4. Validate that the parameters replicate the original system
		
		if(validate):
			jetDict = {"pileup_frac":False, "addresses":addresses, "orientation":True, "boost":True}
			
			prongs_reco = PowerJets.SortByE(PowerJets.GetJets(orientation + boost_params + splitting, jetDict))
			totalE_orig = sum(prongs_orig, Vec4()).x0
			
			# Scale the reconstructed jets to the correct energy
			for jet in prongs_reco:
				jet *= totalE_orig
				
			#~ print()
			#~ for jet in prongs_orig:
				#~ print("{: 1.3e} {: 1.3e} {: 1.3e} {: 1.16e}".format(jet.x0, jet.Eta(), jet.Phi(), jet.Beta()))
				
			#~ print()
			#~ for jet in prongs_reco:
				#~ print("{: 1.3e} {: 1.3e} {: 1.3e} {: 1.16e}".format(jet.x0, jet.Eta(), jet.Phi(), jet.p().Mag() / jet.x0))
				
			#~ print()
			#~ for orig, reco in zip(prongs_orig, prongs_reco):
				#~ diff = orig-reco
				#~ print("{: 1.3e} {: 1.3e} {: 1.16e}".format(diff.x0, diff.p().Mag(), orig.Beta()))
				
			# Verify the absolute orientation to high accuracy
			error_p3 = math.sqrt(sum(map(lambda p1, p2: (p1 - p2).Mag2() / (p1 + p2).Mag2(), 
				map(Vec4.p, prongs_orig), 
				map(Vec4.p, prongs_reco))) / len(prongs_orig))
				
			# Verify the relative orientation to high accuracy
			mass_diff = list(map(lambda m1, m2: ((m1-m2)/(m1+m2))**2, 
				PowerJets.EnsembleMass2(prongs_orig), PowerJets.EnsembleMass2(prongs_reco)))
				
			error_mass = math.sqrt(sum(mass_diff) / len(mass_diff))
			
			#~ print()
			#~ print("error: ", error_p3, error_mass)
			assert(error_p3 < 1e-10)
			assert(error_mass < 1e-10)
				
		return splitting, addresses, orientation, boost_params
		
########################################################################
		
pythia = PowerJets("PowerJets.ini")

while(pythia.Next()):
	pythia.WriteAllVisibleAsTowers("towers.dat")
		
	tracks = sorted(pythia.Get_Tracks(), key = lambda p3: p3.Mag2(), reverse=True)
	towers = sorted(pythia.Get_Towers(), key = lambda p3: p3.Mag2(), reverse=True)
	
	#~ print()
	#~ print("tracks")
	#~ for p3 in tracks:
		#~ print(p3.Mag(), p3.Eta(), p3.Phi())
	
	#~ print()
	#~ print("towers")
	#~ for p3 in towers[0:30]:
		#~ print(p3.Mag(), p3.Eta(), p3.Phi())
	
	jets_raw = pythia.Cluster_FastJets(False)
	jets = pythia.Cluster_FastJets(True)
	
	total_p4 = sum(jets, Vec4())
	
	totalE_raw = sum(jets_raw, Vec4()).x0
	totalE_sub = total_p4.x0
	totalE = 0.
	
	for jet in jets:
		if(jet.x0 > .05 * totalE_sub):
			totalE += jet.x0	
	
	#~ f_PU = (totalE_raw - totalE)/totalE_raw
	f_PU = 0.0
	#~ f_PU = 1e-3
	#~ f_PU =(totalE_raw - 250.)/totalE_raw
	
	
	
	print("index:", pythia.EventIndex(), "beta_z:", total_p4.x3 / total_p4.x0)
	splitting, addresses, orientation, boost = PowerJets.SolveJetParameters(
		jets, f_min=0.05, nProngs=5, validate=True)
		
	jetParamDict = {"pileup_frac":True, "addresses":addresses, "hybrid":False,
		"orientation":True, "boost":True}
		
	bounds_PU = ((0., 1e-5),)
	bounds_orientation = ((0., math.pi), (-2.*math.pi, 2.*math.pi), (-2.*math.pi, 2.*math.pi))
	bounds_boost = ((0., 1.), (-float("inf"), float("inf")), (-2.*math.pi, 2.*math.pi))
	bounds_jets = ((1e-3, 0.999),)*3 + ((-math.pi, math.pi),)
		
	xParams = [f_PU,] + orientation + boost + splitting
	xBounds = bounds_PU + bounds_orientation + bounds_boost + bounds_jets[0:2] + bounds_jets[0:3]
	
	while(len(xBounds) < len(xParams)):
		xBounds += bounds_jets
		
	assert(len(xBounds) == len(xParams))
	
	xParams_orig = copy.deepcopy(xParams)
	
	lMax = 1024
	Hl_ME = pythia.Hl_ME(lMax)
	Hl_Obs = pythia.Hl_Obs(lMax)
	Hl_Jet_orig = pythia.Hl_Jet(xParams_orig, jetParamDict, lMax)
	
	error = 1e-8
	assert(all((x >= -error) and (x <= 1. + error) for x in Hl_ME))
	assert(all((x >= -error) and (x <= 1. + error) for x in Hl_Obs))
	assert(all((x >= -error) and (x <= 1. + error) for x in Hl_Jet_orig))	
	
	#~ Hl_Jet = pythia.Hl_Jet(fit.x, jetParamDict, lMax)
			
	with open("test.dat", "w") as file:
		for lMinus1 in range(lMax):
			file.write("{} {:3e} {:3e} {:3e}\n".format(lMinus1 + 1, 
				Hl_ME[lMinus1], Hl_Obs[lMinus1],
				Hl_Jet_orig[lMinus1]))
				
				
	#~ with open("test.dat", "w") as file:
		#~ for lMinus1 in range(lMax):
			#~ file.write("{} {:3e}\n".format(lMinus1 + 1, Hl_ME[lMinus1]))
			
	continue
			
	print("\nfitting")
	for lMax in [30]:
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
		
	lMax = 256
	
	Hl_Jet_orig = pythia.Hl_Jet(xParams_orig, jetParamDict, lMax)
	Hl_Jet = pythia.Hl_Jet(fit.x, jetParamDict, lMax)
	Hl_Obs = pythia.Hl_Obs(lMax)
	
	print(fit.x)
		
	with open("test.dat", "w") as file:
		for lMinus1 in range(lMax):
			file.write("{} {:3e} {:3e} {:3e}\n".format(lMinus1 + 1, 
				Hl_Obs[lMinus1], Hl_Jet_orig[lMinus1], Hl_Jet[lMinus1]))	
	
	jets_seed = pythia.SortByE(pythia.GetJets(xParams_orig, jetParamDict))			
	jets_reco = pythia.SortByE(pythia.GetJets(fit.x, jetParamDict))	
	jetE = totalE_raw *(1. - fit.x[0])
	#~ jetE = totalE	

	print()
	print(fit.x[0], f_PU)
	print(len(fit.x))

	print()
	print("fit")
	for jet in jets_reco: #pythia.SortByE(PowerJets.ExclusiveCluster(jets_reco, 3)):
		print(jet.x0*jetE, jet.Eta(), jet.Phi())
	
	print()
	print("ME")
	for jet in pythia.SortByE(pythia.Get_ME()):
		print(jet.x0, jet.Eta(), jet.Phi())
		
	#~ print()
	#~ for jet in jets:
		#~ print(jet, jet.Beta())
		
	print()
	print("anti-kt")
	for jet in jets[0:4]:
		print(jet.x0, jet.Eta(), jet.Phi())
	
	print()
	print("seed")
	for jet in jets_seed:
		print(jet.x0*(1.-f_PU)*totalE_raw, jet.Eta(), jet.Phi(), jet.Beta())
	
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
		
	
