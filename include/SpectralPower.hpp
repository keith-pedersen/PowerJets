#ifndef SPECTRAL_POWER
#define SPECTRAL_POWER

/*! @file SpectralPower.hpp
 *  @brief Tools to efficiently calculate the spectral power H_l (i.e. the Fox-Wolfram moments).
 *  @author Copyright (C) 2017 Keith Pedersen (Keith.David.Pedersen@gmail.com)
 *  @date August 2017
*/

#include "PowerJets.hpp"
#include "kdp/kdpVectors.hpp"
#include "pqRand/pqRand.hpp"
//~ #include "NjetModel.hpp"
//~ #include "zLib/kdpRandom.hpp"
#include "SelfOuterU.hpp"
#include "Pythia8/Event.h"
#include <QtCore/QSettings>
#include <mutex>
#include <memory>

class Jet;
class ShapedJet;
class ShapeFunction;

/*! @brief An object for managing parallel calculation of spectral power \f$ H_l \f$.
 *  
 *  SpectralPower calculates the spectral power of a set of N particles
 *  \f[
 *      H_l = (2l + 1) f_i \, P_l( \hat{p}_i \cdot \hat{p}_j ) f_j
 *  \f]
 * 
 *  where:
 *   - Each particle is assumed to be massless.
 *   - \f$ f_i = E_i / E_\text{tot} \f$ is each particle's energy fraction.
 *   - \f$ \hat{p}_i = \vec{p}_i / |\vec{p}_i| \f$ is the unit-direction of each particle's 3-momentum.
 *   - The Legendre polynomial \f$ P_l \f$ independently operates on each scalar product.
 * 
 *  To calculate \f$ H_l \f$, we use the recursive definition of Legendre polynomials
 *  \f[
 *     (l + 1) P_{l+1}(x) = (2l + 1) x \, P_{l}(x) - l \, P_{l-1}(x)
 *  \f]
 *  
 *  Since \f$ \hat{p}_i \cdot \hat{p}_j \f$ is a symmetric matrix, 
 *  we only need the upper half U, which we can pack into a 
 *  1D array \f$ {\tt pDot}_k \f$ (using \ref TiledSelfOuterU).
 *  Equivalently, \f$ f_i f_j \f$ can also be packed in a 1D array \f$ {\tt fProd}_k \f$. 
 *  We then recursively calculate \f$ P_l({\tt pDot}_k) \f$, which permits
 *  \f[
 *    H_l = P_l({\tt pDot}_k)\, {\tt fProd}_k 
 *  \f]
 *  When calculating \f$ P_l \f$ recursively, each element from \f$ {\tt pDot}_k \f$ is 
 *  completely independent. This means that \f$ H_l \f$ can be constructed from the sum of 
 *  many independent pieces, each the inner product of matching subsets of 
 *           \f$ {\tt pDot}_k \f$ and \f$ {\tt fProd}_k \f$
 *  \f[
 *    H_l = \{P_l({\tt pDot}_i)\,{\tt fProd}_i \} + \{P_l({\tt pDot}_j)\,{\tt fProd}_j \} + ...
 *    \qquad\text{ e.g. for } i \in [0, 32] \text{ and } j \in [32, 64]
 *  \f]
 *  This is beneficial purely from a memory standpoint, since there is 
 *  no need to store either \f$ k \f$ array in their entirety (both of size O(N**2)).
 *  The fringe benefit is that the calculation becomes embarrassingly parallel.
*/
class SpectralPower
{
	public:
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t; //!< @brief The 3-vector type
		using vec4_t = PowerJets::vec4_t; //!< @brief The 4-vector type
		
		//! @brief A simple struct for storing individual particle information.
		struct PhatF
		{
			vec3_t pHat; //!< @brief Unit-direction of travel.
			real_t f; //!< @brief Energy fraction (versus total event energy).
			
			//! @brief Construct from p3 (f=|p3>|), normalizing p3 
			PhatF(vec3_t const& p3);
			
			//! @brief Construct from p4 (f=p4.x0), normalizing p3 
			PhatF(vec4_t const& p4);
			
			//! @brief Construct from p4 (f=p4.x0), normalizing p3 
			PhatF(Jet const& jet);
			
			/*! @brief Construct from a 3-vector's components and energy fraction
			 *  
			 *  @param px,py,pz	The components of the 3-vec
			 *  @param f_in 	The energy fraction
			 */ 
			PhatF(real_t const px, real_t const py, real_t const pz, real_t f_in);
			
			//! @brief Construct from a Pythia particle (discarding mass, using energy only)
			PhatF(Pythia8::Particle const& particle);
			
			//! @brief Construct a vector of PhatF, normalizing the total f of the collection
			template<class T>
			static std::vector<PhatF> To_PhatF_Vec(std::vector<T> const& originalVec)
			{
				std::vector<PhatF> convertedVec;
				
				real_t totalE = real_t(0);
				for(auto const& original : originalVec)
				{
					convertedVec.emplace_back(original);
					totalE += convertedVec.back().f;
				}
				
				for(auto& converted : convertedVec)
					converted.f /= totalE;
					
				return convertedVec;
			}
		};

		/*! @brief A collection of PhatF (an object of vectors)
		 * 
		 *  When we take the outer products \f$ \hat{p}_i \cdot \hat{p}_j \f$
		 *  and \f$ f_i f_j \f$, the object-of-vectors paradigm is much faster than the 
		 *  vector-of-objects paradigm (std::vector<PhatF>), because the 
		 *  object-of-vectors allows SIMD vectorization of the scalar product.
		 *  @note We deliberately emulate many of the functions of std::vector
		*/ 
		class PhatFvec
		{
			// Keep x,y,z,f hidden from everyone but SpectralPower
			// (and any of its nested classes, which have the same friend rights as SpectralPower).
			friend class SpectralPower;
			friend class PowerSpectrum;
			
			private:
				std::vector<real_t> x, y, z;
				std::vector<real_t> f;
			
			public:
				PhatFvec() {} //!< @brief Construct an empty PhatFvec.
				~PhatFvec() {}
				
				/*! @brief Convert a std::vector<PhatF> into a PhatVec
				 *  (vector-of-classes to class-of-vectors).
				 * 
				 *  @param orig 	the std::vector<PhatF>
				 *  @param normalize the pHat in the vector-of-classes
				*/ 
				PhatFvec(std::vector<PhatF> const& orig);
								
				// We can use implicit copy and move for ctors and assigment, 
				// because x, y, z, and f all have them defined.
				
				inline size_t size() const {return f.size();}
				void reserve(size_t const reserveSize);
				void clear();
				
				void emplace_back(PhatF const& pHatF);
				
				// Less versatile than insert, but there's no need to define iterators
				void append(PhatFvec const& tail);
				
				//! @brief Join two PhatFvec returned by separate function calls
				static PhatFvec Join(PhatFvec&& first, PhatFvec&& second);
		};
		
		class ShapedParticleContainer : public PhatFvec
		{
			friend class PowerSpectrum;
			
			private:
				/* There are three cases for shapes in the container: 
				 * 1. Every particle has the same shape (i.e. tracks)
				 * 2. Sets of adjacent particles have the same shape (i.e. towers in a 
				 * calorimeter whose bands do not have identical solid angle, as in a hadronic calorimeter)
				 * 3. Every particle has unique shape.
				 * 
				 * We now define a system which is efficient for all three.
				 * 
				 * We keep all shapes in shape_store, copying any incoming shape.
				 * + For case 1, we store only one shape in shape_store.
				 * + For case 2 and 3, we keep a vector of iterators to shapes,
				 *   one for each particle, which allows multiple particles to
				 *   point to the same shape. This reduces 
				 *   redundant calculation of hl during recursion.
				 * Storing the shapes in shape_store improves cache efficiency, 
				 * since adjacent particles will have adjacent shapes.
				*/
				std::vector<ShapeFunction*> shapeStore; // The pointers which need deleting
				std::vector<ShapeFunction*> shapeVec;
				//~ std::vector<decltype(shape_store::iterator)> shape;
				
			public:
				ShapedParticleContainer(std::vector<ShapedJet> const& jets);
			
				ShapedParticleContainer(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape);
				
				~ShapedParticleContainer();
				
				void append(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape);
		};
		
		/*! @brief A thread-safe object for generating increments of the
		 *  self outer product (SOP) \f$ \hat{p}_i \cdot \hat{p}_j \f$ and \f$ f_i f_j \f$
		*/ 
		class Outer_Increment
		{
			public:
				// Must be a power of 2 to use TiledSelfOuterU
				static size_t constexpr tileWidth = 8;
							
				// A bigger tileWidth is better, because each increment's contribution
				// is (extern) summed via binary reduction. A larger incrementSize
				// leads to the accumulation of less floating-point error.
				static size_t constexpr incrementSize = tileWidth * tileWidth;
			
			private:
				// Since these arrays have the same structure, it would be nice to 
				// use a single object to manage the increment overhead. 
				// For right now, it's much easier to use four different objects.
				TiledSelfOuterU_Incremental<real_t, Equals, tileWidth> fOuter;
				TiledSelfOuterU_Incremental<real_t, Equals, tileWidth> xOuter;
				TiledSelfOuterU_Incremental<real_t, PlusEquals, tileWidth> yOuter;
				TiledSelfOuterU_Incremental<real_t, PlusEquals, tileWidth> zOuter;
				
				// We only want one thread at a time calling Setup() or Next(),
				// since this alters the internal state of each of the Outer_Increment objects,
				// and they must remain synced.
				std::mutex syncLock;
				
			public:
				//! @brief The ctor does nothing; use Setup().
				Outer_Increment() {}
					
				/*! @brief Bind to \p source in preparation to compute its SOP,
				 *  but don't calculate anything yet.
				 *  
				 *  @warning No guarantee is made that the last SOP was completed.
				 *  
				 *  @param source		The input PhatFvec.
				 *  @return The number of increments that will be returned by Next().
				*/ 
				size_t Setup(PhatFvec const& source);
				
				/*! @brief Emplace the next increment of the SOP into 
				 *  \p pDot_incr and \p fProd_incr.
				 * 
				 *  When the increment is not full (final few increments),
				 *  \p pDot_incr and \p fProd_incr are padded with zeroes on the right.
				 *  When no increments remain (the SOP was already completed), both are unaltered.
				 *  Hence, the return informs the state of the SOP calculation
				 *  (so that while(obj.Next()) is a good way to make a loop):
				 *  - normal (full) increment ... return == \p incrementSize
				 *  - final (usually partial) increment ... 0 < return <= \p incrementSize
				 *  - SOP done ... return == 0 (\p pDot_incr and \p fProd_inc unaltered)
				 * 
				 *  @param pDot_incr	The next increment of \f$ \hat{p}_i \cdot \hat{p}_j \f$
				 *  @param fProd_incr	The next increment of \f$ f_i f_j \f$
				 *  @return The length of the increment emplaced (not counting zero fill).
				*/ 
				size_t Next(std::array<real_t, incrementSize>& pDot_incr, 
					std::array<real_t, incrementSize>& fProd_incr);
		};
		
		GCC_IGNORE(-Wpadded)
		/*! @brief A settings container for PowerSpectrum
		 * 
		 *  Settings are read from a QSettings object (a parsed INI file),
		 *  where the INI file contains settings of the format "power/variable" 
		 *  (where "variable" is the name of the Settings class member).
		*/
		struct Settings
		{
			size_t maxThreads; //!< Max number of worker threads to use.
			size_t minIncrements; //!< Min number of increments per thread.
			size_t lMax; //!< Calculate H_l from l=1 to lMax.
			
			bool lFactor; //!< Whether to affix (2l + 1) prefix to H_l.
			bool nFactor; //!< Divide H_l by <f|f>, to scale out multiplicity factors.
			
			Settings(QSettings const& parsedINI):
				maxThreads(size_t(parsedINI.value("power/maxThreads", 4).toULongLong())),
				minIncrements(size_t(parsedINI.value("power/minIncrements", 2).toULongLong())),
				lMax(size_t(parsedINI.value("power/lMax", 1024).toULongLong())),
				lFactor(parsedINI.value("power/lFactor", false).toBool()),
				nFactor(parsedINI.value("power/nFactor", false).toBool()) {}
		};
		GCC_IGNORE_END
		
	private:
		Outer_Increment outer; // Manages the generation of SOP increments.
		Settings settings; // Settings for H_l calculation and format.
		
		// Threaded implementation for calculating an increment of H_l
		std::vector<real_t> H_l_threadedIncrement();
		
		static std::vector<real_t> Hl_Extensive_SelfTerm(size_t const lMax,
			std::vector<PhatF> const& particles, std::vector<real_t> const& hl_onAxis);

		static std::vector<real_t> Hl_Extensive_SubTerm(size_t const lMax,
			std::vector<PhatF> const& left, std::vector<real_t> const& hl_onAxis_left, 
			std::vector<PhatF> const& right, std::vector<real_t> const& hl_OnAxis_right);
			
		static std::vector<real_t> Hl_Jets_Particles_SubTerm(size_t const lMax,
			std::vector<ShapedJet> const& jets,  std::vector<real_t> const& hl_onAxis_Filter,
			std::vector<PhatF> const& particles, std::vector<real_t> const& hl_onAxis_particles);
		
	public:
		/*! @brief Read and store the settings.
		 *  
		 *  @param parsedSettings	The settings, already parsed from the INI file.
		*/
		SpectralPower():
			SpectralPower(QSettings()) {}
		
		SpectralPower(QSettings const& parsedSettings):
			settings(parsedSettings) {}
		SpectralPower(std::string const& iniPath):
			SpectralPower(QSettings(iniPath.c_str(), QSettings::IniFormat)) {}
		
		Settings const& GetSettings() {return settings;}
		Settings const& UpdateSettings(QSettings const& parsedSettings);
							
		/*! @brief Calculate and return \f$ H_l \f$ (l = 1 to \p lMax, since \f$ H_0 = 1 \f$ always).
		 * 
		 *  If settings.lFactor, each H_l is scaled by (2l+1).
		 *  If settings.nFactor, each H_l is scaled by 1/<f|f>.
		 * 
		 *  @param input	The vector of particles.
		 *  @param lMax
		 *  @param numThreads_requested	The number of threads requested.
		 *  @return The power spectrum.
		*/ 
		std::vector<real_t> Hl(size_t const lMax,
			PhatFvec const& particles,
			size_t const numThreads_requested = 0); // If zero, defer to internal settings
			
		std::vector<real_t> Hl(size_t const lMax, 
			std::vector<vec3_t> const& particles,			
			size_t const numThreads_requested = 0); // If zero, defer to internal settings
			
		static std::vector<real_t> Hl_Obs(size_t const lMax,
			std::vector<PhatF> const& particles, ShapeFunction& particleShape);
			
		static std::vector<real_t> Hl_Obs(size_t const lMax,
			std::vector<PhatF> const& tracks, ShapeFunction& trackShape, 
			std::vector<PhatF> const& towers, ShapeFunction& towerShape);
		
		static std::vector<real_t> Hl_Jet(size_t const lMax, 
			std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter);
		
		static std::vector<real_t> Hl_Hybrid(size_t const lMax,
			std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter,
			std::vector<PhatF> const& tracks, ShapeFunction& trackShape,
			std::vector<PhatF> const& towers, ShapeFunction& towerShape,
			std::vector<real_t> const& Hl_Obs_in = std::vector<real_t>());
				
		void Write_Hl_toFile(std::string const& filePath,
			std::vector<vec3_t> const& particles,
			size_t const lMax, size_t const numThreads_requested = 0);			
		/* 
		 * 
		 * Producer-consumer model (first back up working single thread model)
		 * Multiple producer threads, one consumer.
		 * producers write to shared H_l accumulator once while loop ends (using write muteX)
		 * producers use separate read mutex built into outer
		*/
};

// Keith's rules of programming

/*
 *	1. Six months from now, you will have forgotten 
 * 	the important details of your code. Therefore: 
 * 		a. Make things private to protect YOU from YOURSELF.
 *				-  Hidden implementation prevents YOU from arrogantly using the 
 * 				object incorrectly, inappropriately, or recklessly.
 * 			-  "We're all adults here" correctly presumes good intentions, 
 * 				but neglects to account for ignorance, negligence, or errors.
 * 			-  Hidden implementation exposes only the API, leaving you free
 * 				to change the internal machinations without bricking existing code.
 * 		b. Comments remind YOU what the hell YOU were thinking six months ago.
 * 			Why is it designed this way? How does this wicked optimization work?
 * 			What the hell is going on?!
 * 2. Don't repeat yourself (DRY) may not always be fastest, but it creates 
 * 	the most reliable code (debugged, validated, DON'T TOUCH).
 * 3. Leave space (in the API) for features or functions you may one day need. 
 * 	BUT DON'T WRITE THEM TILL YOU NEED THEM!
*/ 

#endif
