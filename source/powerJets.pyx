# This is a Cython wrapper for the powerJets C++ package
'''Compute the power spectrum H_l from an n-jet model.

	This package is under active development, and it's API is subject to dramatic changes.
'''

########################################################################
########################################################################
########################################################################
# We first declare the C++ functions we intend to call
########################################################################

#~ from libc.stdint cimport int64_t
#~ from libcpp cimport bool # access bool as bool
from libcpp.string cimport string # access std::string as string
#~ from libc.string cimport memcpy
from libcpp.vector cimport vector # access std::vector<T> as vector[T]
from cython.operator cimport dereference as deref 
from libcpp.algorithm cimport sort as stdsort
# Dereference Cython pointers via deref() (e.g. func(deref(pointer)),
# to pass an object as argument. Note: pointer.func() has implicit dereference)

import numpy
cimport numpy

import math

#####################################################################

cdef StdVecToNumpy(const vector[double]& stdVec):
	# Using this "C API" for numpy is deprecated, as the compiler warns,
	# but I don't know any alternative
	cdef numpy.ndarray[double, ndim = 1, mode="c"] numpyVec = numpy.empty(stdVec.size())
	cdef double* vec = <double*>numpyVec.data
	
	# The memcpy appears to the be no faster than the loop, and is less readable
#~ 	memcpy(<void*>vec, <void*>stdVec.data(), stdVec.size()*sizeof(real_t))
	for i in range(stdVec.size()):
		vec[i] = stdVec[i]
	return numpyVec
	
#####################################################################

import kdp
cimport kdp

import scipy.optimize

include "kdp/kdpVectors.hpy"
#~ include "/home/keith/Documents/Projects/libkdp/source/kdp.pyx"

#~ cimport kdp

########################################################################
# import pqRand.hpp

# To allow the objects to have the same name in Cython and C++,
# we import as **_c, giving the full C++ name in quotes
cdef extern from "NjetModel.hpp":
	cdef cppclass Jet:
		Vec4_c p4
		double mass
		
		@staticmethod
		void Rotate[T](vector[T]& jetVec, const double theta, const double phi, const double omega)
		
	cdef cppclass ShapedJet:
		Vec4_c p4
		double mass
		vector[bool] address
		
		ShapedJet()
		ShapedJet(const double, const double, const double, const double, 
			const Vec4from2, vector[bool], const vector[double]&) except +
		ShapedJet(const Vec3_c&, const double, const Vec4from2, vector[bool], const vector[double]&) except +
		
		bool operator<(const ShapedJet& that)
		
#~ 		@staticmethod
#~ 		bool Sort_by_Mass(const ShapedJet& left, const ShapedJet& right)
		
	cdef cppclass ShowerParticle:
		# Because we want to catch and return C++ exceptions (except+),
		# Cython requires a nullary constructor due to its automatic coding paradigm.
		# default assignment (shallow copy) of a binary tree can create pointer errors.
		# There are two solutions. a = b is equivalent to std::swap(a, b), 
		# or switch to shared_ptr for b and c. 
		ShowerParticle(const vector[double]&, const vector[vector[bool]]&) except +
		
		vector[ShapedJet] GetJets() 
		
	cdef cppclass NjetModel_c "NjetModel":
		# The nested class for internal reuse by H_l_JetParticle
		cppclass JetParticle_Cache:
			JetParticle_Cache()
			size_t lMax() const
		
		NjetModel_c(const string& iniFileName) except +
		
		JetParticle_Cache Make_JetParticle_Cache(const vector[ShapedJet]& jetVec_in,
			const size_t lMax, const double jetShapeGranularity) except +
		
		vector[double] H_l(vector[ShapedJet]& const, size_t const, double const) except +
		vector[double] H_l_JetParticle(const JetParticle_Cache& cache, 
			const vector[PhatF]& particles, 
			const double theta, const double phi, const double omega) except +
		
		
cdef extern from "SpectralPower.hpp":
	cdef cppclass PhatF "SpectralPower:PhatF":
		pass	
		
cdef extern from "LHE_Pythia_PowerJets.hpp":
	cdef cppclass Status "LHE_Pythia_PowerJets::Status":
		bool operator==(Status const)
		
	cdef Status Status_OK "LHE_Pythia_PowerJets::Status::OK"
	cdef Status Status_UNINIT "LHE_Pythia_PowerJets::Status::UNINIT"
	cdef Status Status_END_OF_FILE "LHE_Pythia_PowerJets::Status::END_OF_FILE"
	cdef Status Status_EVENT_MAX "LHE_Pythia_PowerJets::Status::EVENT_MAX"
	cdef Status Status_ABORT_MAX "LHE_Pythia_PowerJets::Status::ABORT_MAX"
			
	cdef cppclass LHE_Pythia_PowerJets:
		LHE_Pythia_PowerJets(const string&) except +
		
		size_t EventIndex()		
		Status GetStatus()
		
		const vector[Vec3_c]& Get_Detected()	
		const vector[PhatF]& Get_Detected_PhatF()
		const vector[double]& Get_H_det()		
		const vector[Jet]& Get_FastJets()
		const vector[Jet]& Get_ME()
		
		Status Next()
			
########################################################################		
# Now we make the Cython wrapper class of the C++ object

########################################################################		
# Python 3 uses unicode str; Python 2 uses ASCII strings; C++ std::string is ASCII only.
# Thus, Python 2 is a naive conversion between std::string and str, 
# whereas Python 3 needs to explicitly "encode"/"decode" the unicode.
# To allow this package to work with either Python version, we define two hidden helper functions
import sys

if(sys.version_info < (3,0)):
	def _str2string(s):
		return s
	def _string2str(s):
		return str(s)
else:
	def _str2string(s):
		return s.encode()
	def _string2str(s):
		return s.decode()
	
#####################################################################	

from enum import Enum
class JetParamStyle(Enum):
	Momentum = 1
	Splitting = 2

#####################################################################

from enum import Enum	
class JetMomentumStyle(Enum):
	Massless = 1
	Direct = 2

#####################################################################
	
pileup_key = "pileup_frac"

# Cython default initializes the C++ object using the default ctor
# Howerver, engine() causes the default seeding, which we do not always want.
# To ellude the auto-default behavior, we must define __cinit__ and __dealloc__, 
# which in this case requires storing a pointer to the C++ object

cdef class _PowerJetFitter:
	''' This class is the compiled innards of a pure Python class we define later.
	The purpose of this class is to translate fit parameters from Python to C
	using compiled C code. It does not actually do any of the fitting.''' 
	cdef LHE_Pythia_PowerJets* c_pythia
	cdef NjetModel_c* c_model
	cdef NjetModel_c.JetParticle_Cache cache
	
	#####################################################################
	
	def __cinit__(self, str iniFileName = "./PowerJets.conf"):
		self.c_pythia = new LHE_Pythia_PowerJets(_str2string(iniFileName))
		self.c_model = new NjetModel_c(_str2string(iniFileName))
	
	#####################################################################	
	
	def __init__(self, str iniFileName):
		'''In order to inherit from this class by calling super().__init__(args) 
			in the __init__() of the inherited class, we need to create this dummy function.'''
		pass
	
	#####################################################################
	
	def __dealloc__(self):
		del self.c_pythia
		del self.c_model
			
	#####################################################################
	
	def Next(self):
		return (self.c_pythia.Next() == Status_OK)
		
	#####################################################################
	def EventIndex(self):
		return self.c_pythia.EventIndex()
		
	#####################################################################
	def Get_H_det(self):
		return StdVecToNumpy(self.c_pythia.Get_H_det())
		
	#####################################################################
	
	@staticmethod
	cdef vector[ShapedJet] _GetJets_Momentum(numpy.ndarray[double, ndim=1] jetParams, dict jetParamDict, 
		const int nShapeParams) except+:
		''' Interpret the jet parameters as jet momenta.'''
		
		cdef int numParams_perJet = 4 + nShapeParams
			
		balancingJet = jetParamDict.get('force_CM', True)
		cdef int maxParams = len(jetParams)
		if(balancingJet): 
			maxParams -= 1
		
		momentumStyle = jetParamDict.get('p3_style', JetMomentumStyle.Massless)
		# JetMomentumStyle
		# 		massless:		[p1, p2, p3, gamma] 	(the momentum if the jet is massless)
		# 		direct:		[p1. p2, p3, mass] 	(the momentum is supplied directly)
		# If balancing jet, the final parameter follows the mass convention
						
		cdef vector[ShapedJet] jetVec
		cdef Vec4_c total	
		cdef Vec4from2 style = V4f2_Boost_preserve_E if (momentumStyle == JetMomentumStyle.Massless) else V4f2_Mass
						
		if balancingJet:
			if((len(jetParams) % numParams_perJet) != 1):
				raise ValueError("PowerJetFitter.GetJets: Momentum basis. Looking for 4 params per jet + mass/boost of balancing jet")
				
		elif ((len(jetParams) % numParams_perJet) != 0):
			raise ValueError("PowerJetFitter.GetJets: Momentum basis. Looking for 4 params per jet (and no balancing jet)")
		
		for i in range(0, maxParams, numParams_perJet):
			jetVec.push_back(ShapedJet(
				jetParams[i + 0], jetParams[i + 1], jetParams[i + 2],
				jetParams[i + 3], style, 
				vector[bool](), jetParams[i + 4: i + 4 + nShapeParams]))
				
			total += jetVec.back().p4
			
		if balancingJet:
			# For both massless and direct momenta, balancing jet's mass is parameterized by gamma,
			# though this time it preserves momentum. For now, this jet has no shape.
			jetVec.push_back(ShapedJet(-total.p(), jetParams[-1], V4f2_Boost_preserve_p3, 
				vector[bool](), vector[double]()))
			
		return jetVec
			
	#####################################################################
	
	@staticmethod
	cdef vector[ShapedJet] _GetJets_Splitting(numpy.ndarray[double, ndim=1] jetParams, dict jetParamDict, 
		const int nShapeParams) except+:
		'''Interpret the jet parameters as a splitting tree.'''
		
		return ShowerParticle(jetParams, jetParamDict.get("addresses", list())).GetJets()
			
	#####################################################################
	
	@staticmethod
	cdef vector[ShapedJet] _GetJets(numpy.ndarray[double, ndim=1] jetParams, dict jetParamDict) except+:
		'''The internal GetJets, which deals with the pileup fractioni and 
		switches between the parameter style.'''	
		
		######################
		# discard jet fraction (I believe this does not alter the argument, but I'm not totally sure)
		if (jetParamDict.get(pileup_key, True)):
			jetParams = jetParams[1:]
			
		#######################
		# determine param style
		paramStyle = jetParamDict.get('param_style', JetParamStyle.Splitting)
		
		cdef int nShapeParams = jetParamDict.get('num_shape_params', 0)
		if(nShapeParams < 0):
			raise ValueError("PowerJetFitter.GetJets: num_shape_params must be non-negative")		
		
		if(paramStyle == JetParamStyle.Momentum):
			return _PowerJetFitter._GetJets_Momentum(jetParams, jetParamDict, nShapeParams)
		elif(paramStyle == JetParamStyle.Splitting):
			return _PowerJetFitter._GetJets_Splitting(jetParams, jetParamDict, nShapeParams)
		else:
			raise ValueError("PowerJetFitter: unrecognized 'param_style' in jetParamDict")		
			
	#####################################################################
	
	def Get_FastJets(self):
		# Cython can't do a const references, must copy; stupid
		cdef vector[Jet] jetVec = self.c_pythia.Get_FastJets()
		
		return [kdp.Vec4.Factory(<const Vec4_c&>jet.p4) for jet in jetVec]
		
	def Get_ME(self):
		# Cython can't do a const references, must copy; stupid
		cdef vector[Jet] jetVec = self.c_pythia.Get_ME()
		
		return [kdp.Vec4.Factory(<const Vec4_c&>jet.p4) for jet in jetVec]
	
	def SetupOrientationFit(self, jetParams, dict jetParamDict, int lMax=128, double granularity=1e6):
		'''The orientation fit will use the same jets and density rho'''
		self.cache = self.c_model.Make_JetParticle_Cache( 
			_PowerJetFitter._GetJets(numpy.asarray(jetParams), jetParamDict),
			lMax, granularity)
	
	@staticmethod
	def GetJets_All(jetParams, dict jetParamDict, 
		const double rotate_theta = 0., const double rotate_phi = 0., const double rotate_omega = 0.):	
		# Use the internal function to get a vector[Jet], then copy the 4-vectors into newly minted Vec4
		cdef vector[ShapedJet] jetVec = _PowerJetFitter._GetJets(numpy.asarray(jetParams), jetParamDict)
		
		if((rotate_theta != 0.) and (rotate_phi != 0.)):
			Jet.Rotate(jetVec, rotate_theta, rotate_phi, rotate_omega)
		
		return [kdp.Vec4.Factory(<const Vec4_c&>jet.p4) for jet in jetVec], [jet.mass for jet in jetVec], [jet.address for jet in jetVec]
	
	@staticmethod
	def GetJets(jetParams, dict jetParamDict, const double rotate_theta = 0., const double rotate_phi = 0.):
		''' Get the jets produced by a set of parameters and their dictionary, 
		returning their 4-vectors as Vec4.'''
					
		# I don't know why, but for some reason Cython complains that it can't 
		# convert from Vec4_c to const Vec4_c, so we have to cast the const-ness ... weird
		
		jets, _, _ = _PowerJetFitter.GetJets_All(jetParams, jetParamDict, rotate_theta, rotate_phi)
		return jets
		
	@staticmethod
	def GetJets_Rest_Sorted(jetParams, dict jetParamDict):
		''' Get the jets produced by a set of parameters and their dictionary, 
		returning all their mass and address (as separate returns).
		The addressses only make sense in a splitting tree.'''
	
		cdef vector[ShapedJet] jetVec = _PowerJetFitter._GetJets(numpy.asarray(jetParams), jetParamDict)
		
		stdsort(jetVec.begin(), jetVec.end());
		
		boostList = list()
		addressList = list()
		
		for jet in jetVec:
			boostList += [jet.p4.x0 / jet.mass,]
			addressList += [tuple(jet.address),]
			
		return boostList, addressList
	
	#####################################################################
	
	# Formerly __call__, the operator() in C++
	cpdef H_l(self, jetParams, dict jetParamDict = {}, int lMax=128, double granularity=1e3):
		''' Given jetParams in the correct format (see GetJets),
			return the power specturm H_l_jet.
			
			jetShapeGranularity is the approximate number of random numbers drawn.'''
		if(lMax < 0):
			raise ValueError("PowerJetFitter(): lMax must be non-negative.")
		
		jetFraction = 1.
		if (jetParamDict.get(pileup_key, True)):
			jetFraction -= jetParams[0]
			
		return StdVecToNumpy(self.c_model.H_l(_PowerJetFitter._GetJets(numpy.asarray(jetParams), jetParamDict), 
			lMax, granularity)) * jetFraction**2
			
	#####################################################################
	
	@staticmethod
	cdef numpy.ndarray[double, ndim=1] _RelErrorUtil(
		numpy.ndarray[double, ndim=1] fit, numpy.ndarray[double, ndim=1] obs):
		# The relative error, but with a utility function that discards
		# high relative errors from tiny values.
		# We must still be careful, for if obs = 0, we get (inf * 0 = nan)
		# (this is not theoretical, it happened often enough for me to track it down)
		return (fit/(numpy.fabs(obs) + 1e-16) - 1.)*numpy.tanh(obs*10.)
		
	@staticmethod
	cdef numpy.ndarray[double, ndim=1] _AbsErrorUtil(
		numpy.ndarray[double, ndim=1] fit, numpy.ndarray[double, ndim=1] obs):
		# The relative error, but with a utility function that discards
		# high relative errors from tiny values.
		return fit - obs
		
	@staticmethod
	cdef numpy.ndarray[double, ndim=1] _H_l_error_vec(
		numpy.ndarray[double, ndim=1] fit, numpy.ndarray[double, ndim=1] obs, 
		bool relative = False):
			
		cdef numpy.ndarray[double, ndim=1] resid
		
		if relative:
			resid = _PowerJetFitter._RelErrorUtil(fit, obs)
		else:
			resid = _PowerJetFitter._AbsErrorUtil(fit, obs)
			
		return resid/numpy.power(numpy.arange(1, len(fit) + 1), 0.25)
			

	def H_l_jet_error_vec(self, jetParams, dict jetParamDict, int lMax, double granularity, 
		const bool relative = False):
		''' Given jetParams in the correct format (see GetJets),
			return the vector of residuals (chi_l = H_l_jet - H_l_obs).
			
			granularity is the approximate number of random numbers drawn.
			
			If relative = True, the residuals are relative.'''
			
		H_l_fit = self.H_l(jetParams, jetParamDict, lMax, granularity)
			
		return _PowerJetFitter._H_l_error_vec(H_l_fit, self.Get_H_det()[0:lMax], relative)
			
	def H_l_jet_particle_error_vec(self, angleParams, numpy.ndarray[double, ndim=1] H_obs,
		const double pileupFrac = 0., const bool relative = False):
		
		H_l_fit = StdVecToNumpy(self.c_model.H_l_JetParticle(self.cache, 
			self.c_pythia.Get_Detected_PhatF(), 
			angleParams[0], angleParams[1], angleParams[2]))*(1. - pileupFrac)**2
		
		return _PowerJetFitter._H_l_error_vec(H_l_fit, H_obs, relative)
