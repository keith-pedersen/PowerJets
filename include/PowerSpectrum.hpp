#ifndef POWER_SPECTRUM
#define POWER_SPECTRUM

// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PowerJets.hpp"
#include "NjetModel.hpp"

#include "kdp/kdpTools.hpp"
#include "kdp/TiledOuter.hpp"

#include "Pythia8/Event.h"

#include <memory> // shared_ptr
#include <atomic>
#include <condition_variable>
#include <thread>
#include <memory>

GCC_IGNORE_PUSH(-Wpadded)

/*! @file PowerSpectrum.hpp
 *  @brief Defines a class that manages a thread pool to efficiently calculate
 *  the dimensionless, angular power spectrum H_l.
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/

//======================================================================
// Homo sapiens
//                      
//      ()      5 - head:   and the way.
//     /||\     4 - heart:  but you have the will
//    / || \    3 - hands:  It will need maintenance;
//     //\\     2 - heat:   you'll find your home.
//    //  \\    1 - feet:   After some searching,
//======================================================================

/*! @brief A class managing parallel, SIMD (on CPU), cache-efficient calculation
 *  of the dimensionless, angular power spectrum of a collection of \f$ N \f$ particles.
 * 
 *  This power spectrum is expressed as "moments" of integer \f$ \ell \f$:
 * 
 *  \f[ H_\ell = \sum_{i,j=1}^N(f_i\,{{}\bar{h}_{(i)}}_\ell)\,P_\ell(\hat{p}_i\cdot\hat{p}_j)\,(f_j\,{{}\bar{h}_{(j)}}_\ell) \f]
 * 
 *  where:
 * 
 *   - \f$ f_i \equiv E_i / E_{\rm tot} \f$ is particle i's energy fraction.
 * 
 *   - \f$ \hat{p}_i \f$ is particle i's direction of travel (a unit vector).
 * 
 *   - \f$ h_{(i)}(\hat{r}) \f$ is particle i's <em> shape function </em>
 *     (its spatial probability distribution, 
 *     azimuthally symmetric about its direction of travel,     
 *     as a function radial position on the unit sphere \f$ \hat{r} \f$).
 * 
 *   - \f$ {{}\bar{h}_{(i)}}_\ell \f$ is the shape function's \f$ \ell \f$-dependent 
 *     "up" coefficient.
 * 
 *   - \f$ P_\ell \f$ is a Legendre polynomial  (which acts independently on each element of 
 *     the matrix of inter-particle vector dot products \f$ \hat{p}_i\cdot\hat{p}_j \f$).
 * 
 *  The specific form and meaning of each of these pieces is explained
 *  in detail in "Harnessing the global correlations of the QCD power spectrum."
 * 
 *  Luckily, the sum over i and j for each moment \f$ \ell \f$ is ultimately collapsed to 
 *  the single scalar \f$ H_\ell \f$, allowing the calculation to be 
 *  broken into smaller \ref Tiles that can be calculated in parallel and accumulated in any order.
 *  Nonetheless, each sub-calculation is rather involved, 
 *  and must be handled with several classes:
 * 
 *   - PhatF is a simple container to store a particle's
 *     direction of travel \f$ \hat{p} \f$ and energy fraction \f$ f \f$.
 * 
 *   - RecursiveLegendre_Increment calculates \f$ P_\ell \f$ recursively for a
 *     "tile" of \f$ \hat{p}_i\cdot\hat{p}_j \f$ (flattened to a 1D array).
 *     Recursion is the simplest way to calculate Legendre polynomials numerically,
 *     especially since we need all \f$ H_\ell \f$ from
 *     \f$ \ell = 1\f$ to \f$ \ell_{\max} \f$. Recursive \f$ P_\ell \f$
 *     also tends to be numerically stable, and use the least FLOPS.
 * 
 *   - ShapeFunction provides a polymorphic interface for several shape 
 *     functions (such as h_Gaussian, h_Boost, h_Cap), whose "up" coefficients 
 *     can be accessed via their respective definitions of ShapeFunction::hl.
 *     In general, these coefficients are also calculated recursively, 
 *     since they rely on the shape function integral
 *     \f$ \bar{h}_\ell = \int_{-1}^{1} {\rm d}z\,h(z) P_\ell(z) \f$, 
 *     whose closed form solution is usually a recurrence relation.
 * 
 *  PowerSpectrum wraps all these pieces into the parallel calculation. 
 *  For efficiency, each PowerSpectrum object manages a thread pool that is 
 *  thread safe, with the first \f$ H_\ell \f$ requested being returned first (FIFO).
 *  Deconstruction will block until all active jobs have finished and returned.
 *  
 *  \note PowerSpectrum is a useful template for the following concepts in 
 *  high-performance computing:
 *  - Multi-threading (thread safety via \c std::mutex).
 *  - Asynchronous calls to thread-safe functions using std::async (see Hl_Hybrid)
 *  - Inter-thread communication (wait and notify via \c std::condition_variable).
 *  - A thread pool (at least twice as efficient as creating new threads for every job).
 *  - \ref Tiles (cache-efficient linear algebra by breaking the job into squares).
 *  - Auto-vectorization on the CPU (SIMD instructions via specially designed loops and data structures).
 *      - e.g. the object-of-vectors paradigm (versus the vector-of-objects).
 *  - Binary accumulation (take CARE of Cancellation And Rounding Error in large sums).
 *  - Hidden implementation (private data structures can change without altering the API).
 *  - Move semantics (request auto-generated move semantics when class members have them).
 */
class PowerSpectrum
{
	public: 
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
				
		//////////////////////////////////////////////////////////////////
		/*! \defgroup Tiles
		 *  @brief The outer products used to calculate \f$ H_l \f$ will be 
		 *  specified and dispatched via a set of tiles.
		 * 
		 *  Details about the superiority of tiling are explained in TiledOuter.hpp
		*/ 
				
		// In the future, these can be dynamic quantities, 
		// but we don't need that now so we'll hard-code them.
		
		//! @brief Max threads in the thread pool
		static constexpr size_t maxThreads = 4;
		
		//! @brief Minimum tiles per thread (to determine how many threads to wake up) \ingroup Tiles
		static constexpr size_t minTilesPerThread = 1; 
		
		/*! \ingroup Tiles
		 *  @brief The width of each square tile 
		 * 
		 * tileWidth should be a power of two and hard-coded for compiler optimization.
		 * 16 is a good number; not too big, not too small.
		 * 8 and 64 are each 40% slower on my machine.
		*/
		static constexpr size_t tileWidth = 16;
		
		//////////////////////////////////////////////////////////////////
		//! \defgroup ParticleContainers
		//! @brief PowerSpectrum uses several classes to communicate particle information
		
		/*! \ingroup ParticleContainers
		 *  @brief A simple struct storing the bare-bones particle information used to calculate Hl
		 * 
		 *  \note Particles can be given a delta-distribution shape (h_Delta)
		 *  if no other shape is supplied (which is how Fox and Wolfram defined 
		 *  their power spectrum); hence, \c pHat and \c f are the minimal information.
		*/ 
		struct PhatF
		{
			/*! @brief The particle's normalized direction of travel (i.e., <tt> pHat.Mag() == 1 </tt>)
			 * 
			 *  \warning All ctors normalize \c pHat, but the user has 
			 *  direct access, and must be careful not to alter its length (we're all adults here).
			*/ 
			vec3_t pHat;
			
			/*! @brief Energy fraction relative to total detected energy (\c f should be in [0, 1]).
			 * 
			 *  \warning No ctors normalize \c f, since that requires access to 
			 *  the entire particle ensemble. The incoming particle energy is 
			 *  treated as \c f, and it is the user's responsibility to 
			 *  properly normalize the ensemble.
			*/ 
			real_t f;
			
			PhatF() = delete;
			
			//! @brief Construct from 3-momentum (<tt> pHat = p3.Normalize() </tt>, <tt> f = p3.Mag() </tt>)
			PhatF(vec3_t const& p3);
			
			//! @brief Construct from 4-momentum (<tt> pHat = p4.p().Normalize() </tt>, <tt> f = p4.x0</tt>)
			PhatF(vec4_t const& p4);
			
			//! @brief Construct from Jet (use 4-momentum ctor)
			PhatF(Jet const& jet);
			
			/*! @brief Construct from a 3-momentum's components and energy fraction
			 *  
			 *  @param px,py,pz	The components of the 3-momentum
			 *  @param f_in 	The energy fraction
			 */ 
			PhatF(real_t const px, real_t const py, real_t const pz, real_t f_in);
			
			//! @brief Construct from a Pythia particle (discarding mass, using 3-momentum and energy)
			PhatF(Pythia8::Particle const& particle);
		};
		
		/*! @ingroup ParticleContainers
		 *  @brief An abstract interface for a collection of PhatF, shared by 
		 *  the vector-of-objects (VecPhatF) and the object-of-vectors (PhatFvec)
		 * 
		 *  This class provides common methods, without specifying the storage/access scheme
		*/ 
		class ParticleContainer
		{
			protected:
				//! @brief The Magnitude of the ith particle's direction of travel
				virtual real_t pHatMag(size_t const i) const = 0;
			
			public:
				void NormalizeF(); //!< @brief Normalize the total energy fraction \p f of the vector to 1.
			
				//! @brief Divide the energy fraction \p f of every object by \p total_f
				virtual void NormalizeF_to(real_t const total_f) = 0;
				
				/*! @brief Verify that fTotal ~= 1 and that all pHat are unit vectors
				 * 
				 *  \param threshold 	\c fTotal and \c pHat.Mag() are normalized
				 *                      when |value-1| < threshold (this accounts for rounding error)
				*/ 
				bool IsNormalized(real_t const threshold = real_t(1e-15)) const;
				
				virtual size_t size() const = 0; //!< @brief The size of the ensemble
				
				//! @brief The total energy fraction \p f of the ensemble
				virtual real_t fTotal() const = 0; 
				//! @brief \f$ \langle f|f \rangle \f$: The total \em square energy fraction \p f**2 of the ensemble
				virtual real_t fInner() const = 0;				
		};
				
		/*! \ingroup ParticleContainers
		 *  @brief A simple wrapper for a std::vector<PhatF> (a vector-of-objects), 
		 *  with several useful utility functions.
		*/ 
		class VecPhatF : public ParticleContainer, public std::vector<PhatF>
		{
			protected:
				virtual real_t pHatMag(size_t const i) const final;
				
			public:
				VecPhatF() = default;
				
				/*! @brief Construct a vector of PhatF from a vector of type T,
				 *  where T can be converted to PhatF.
				 * 
				 *  If \p normalize, call Normalize() on the constructed vector
				*/ 
				template<class T>
				VecPhatF(std::vector<T> const& original, bool const normalizeF = false):
					std::vector<PhatF>(original.cbegin(), original.cend())
				{
					if(normalizeF) NormalizeF();
				}
				
				// We have to manually declare this mapping, because the compiler can't figure it out
				virtual size_t size() const final {return std::vector<PhatF>::size();}
				
				//! @brief Divide the energy fraction \p f of every object by \p total_f
				virtual void NormalizeF_to(real_t const total_f) final;
										
				virtual real_t fTotal() const final; //!< @brief The total energy fraction \p f of the ensemble
				virtual real_t fInner() const final; //!< @brief <f|f>: The total \em square energy fraction \p f**2 of the ensemble
				
				/*! @brief Add particle.f to the running sum and return
				 * 
				 *  This function is supplied as the third argument to std::accumulate by fTotal()
				*/ 
				static real_t Accumulate_f(real_t const sum, PhatF const& particle);
				
				/*! @brief Add (particle.f)**2 to the running sum and return
				 * 
				 *  This function is supplied as the third argument to std::accumulate by fInner()
				*/
				static real_t Accumulate_f2(real_t const sum, PhatF const& particle);
		};
		
		/*! \ingroup ParticleContainers
		 *  @brief A collection of PhatF (an object-of-vectors) for use by PowerSpectrum
		 * 
		 *  Consider two ways of storing a collection of PhatF, each the other's transpose
		 *  (using {} to denote vectors and () for an object's variables):
		 *   - vector-of-objects: {((px_0, py_0, pz_0), f_0), ((px_1, py_1, pz_1), f_1), ...} 
		 *   - object-of-vectors: ({px_0, px_1, ...}, {py_0, py_1, ...}, {pz_0, pz_1, ...}, {f_0, f_1, ...})
		 * 
		 *  When PowerSpectrum calculates the outer products \f$ \hat{p}_i \cdot \hat{p}_j \f$
		 *  and \f$ f_i f_j \f$, the object-of-vectors paradigm is much faster than the 
		 *  vector-of-objects paradigm because the object-of-vectors allows 
		 *  easy SIMD vectorization of the 3-vector dot-product, whose sum:
		 * \verbatim
		    dot_ij = px[i] * px[j] + py[i] * py[j] + pz[i] * pz[j]
		    \endverbatim
		 *  can be broken into three separate loops to accumulate the x, y, and z products, like so:
		 *  \verbatim
		    for(size_t i = 0; i < 16; ++i)
		       for(size_t j = 0; j < 16; ++j)
		          dot[16*i + k] += vec_x[i] * vec_x[j]
		    \endverbatim
		 *  Since adjacent \c j indices have adjacent memory locations,
		 *  the inner loop can be auto-vectorized by the compiler.
		 *  The equivalent loop for the vector-of-objects paradigm cannot be auto-vectorized;
		 *  using SIMD instructions would require hand-coding them.
		 *  Who wants to mess with that? Not me.
		 *  
		 *  \note Since this object is designed to expedite PowerSpectrum's calculations, 
		 *  it does not provide access to its elements (use a VecPhatF for such access). 
		 *  The four independent vectors are kept private so that the can be 
		 *  assured to have the same length.
		*/ 
		class PhatFvec : public ParticleContainer
		{
			// Only the main calculating classes need access.
			friend class PowerSpectrum;
			friend class SpectralPower;
			
			protected:
				std::vector<real_t> x; //!< @brief A vector storing the ensemble's \c pHat.x1
				std::vector<real_t> y; //!< @brief A vector storing the ensemble's \c pHat.x2
				std::vector<real_t> z; //!< @brief A vector storing the ensemble's \c pHat.x3
				std::vector<real_t> f; //!< @brief A vector storing the ensemble's \c f
				
				void reserve(size_t const reserveSize); //!< @brief call vector::reserve on each vector
				void emplace_back(PhatF const& pHatF); //!< @brief Verbatim emplace_back a pHatF
				
				virtual real_t pHatMag(size_t const i) const final;
			
			public:
				PhatFvec() = default;
				~PhatFvec() {}
				
				/*! @brief Convert a std::vector<PhatF> into a PhatVec
				 *  (vector-of-objects to object-of-vectors).
				 * 
				 *  \warning a verbatim copy (\em assumes all \c pHat and \c f are properly normalized).
				*/ 
				PhatFvec(std::vector<PhatF> const& orig);
								
				/* Copy and move semantics are well-defined for the internal vectors, 
				 * and the compiler automatically generates a copy ctor and copy assignment.
				 * But to get move semantics (instead of dumb copy), we have to ask for them.
				*/ 
				PhatFvec(PhatFvec&&) = default;
				PhatFvec& operator=(PhatFvec&&) = default;
				
				size_t size() const {return f.size();}
				
				//! @brief Divide the energy fraction \p f of every object by \p total_f
				virtual void NormalizeF_to(real_t const total_f) final;
				
				virtual real_t fTotal() const final; 
				virtual real_t fInner() const final;
				
				// TODO: remove this vestigial code
				//~ void clear();
				
				// Less versatile than insert, but there's no need to define iterators
				//~ void append(PhatFvec const& tail);
				
				//! @brief Join two PhatFvec returned by separate function calls
				//~ static PhatFvec Join(PhatFvec&& first, PhatFvec&& second);
		};
		
		/*! \ingroup ParticleContainers
		 *  @brief An extension of PhatFvec, adding a ShapeFunction for each particle.
		 * 
		 *  This is the most general tool for calculating the power spectrum
		 *  using PowerSpectrum (which currently only supports 
		 *  azimuthally symmetric shape functions).
		 */ 
		class ShapedParticleContainer : public PhatFvec
		{
			friend class PowerSpectrum;
			friend class LHE_Pythia_PowerJets;
			
			private:
				/* There are three cases for shapes in the container: 
				 * 
				 *  1. Every particle has the same shape (e.g., and enesmble of tracks)
				 * 
				 *  2. Sets of adjacent particles have the same shape 
				 *     (e.g., towers from the same equatorial bands in the calorimeter, 
				 *      which therefore cover the same solid angle).
				 * 
				 *  3. Every particle has unique shape (e.g., an ensemble jets). 
				 * 
				 * Because ShapeFunction objects are not thread safe, 
				 * they will always be cloned before being used inside PowerSpectrum.
				 * Therefore, the easiest way to accommodate all three scenarios
				 * is simply to store pointers to each particle's ShapeFunctions, 
				 * one pointer for each particle (repeated shapes having repeated pointers).
				 * Repeated pointers to shared shapes is beneficial because 
				 * ShapeFunction.hl operates recursively, and remembers its last value. 
				 * So the first time hl(10) is called, math is done, 
				 * but the second time hl(10) is called, it returns the cached value.
				 * However, using pointers this has the nasty side effect that
				 * externally supplied ShapeFunction objects must not die or move.
				 * A safe choice (with automatic garbage collection) is use std::shared_ptr<ShapeFunction>.
				 * ShapeFunction::Make builds a shared_ptr<ShapeFunction> given the supplied arguments,
				 * and ShapeFunction::Clone clones ShapeFunction into a shared_ptr<ShapeFunction>.s
				*/
				std::vector<std::shared_ptr<ShapeFunction>> shapeVec;
								
				// The default shape is a delta-distribution, and the class only needs one
				static const std::shared_ptr<ShapeFunction> delta;
				
			public:
				ShapedParticleContainer() {}
			
				/*! @brief Create an ensemble of extensive jets
				 * 
				 *  If \p normalizeF, normalize the ensemble's energy fraction
				*/ 
				ShapedParticleContainer(std::vector<ShapedJet> const& jets,
					bool const normalizeF = true);
					
				/*! @brief Emplace particles all sharing the same shape, 
				 *  with auto-normalization to their total energy.
				*/
				template<class T> 
				ShapedParticleContainer(std::vector<T> const& particles, 
					std::shared_ptr<ShapeFunction> const& theirSharedShape = delta):
				ShapedParticleContainer(VecPhatF(particles, true), theirSharedShape) // normalizeF = true
				{}
				
				/*! @brief Emplace particles all sharing the same shape, 
				 *  with NO normalization (verbatim copy).
				 */
				ShapedParticleContainer(std::vector<PhatF> const& particles, 
					std::shared_ptr<ShapeFunction> const& theirSharedShape = delta);
					
				// Note; to get move semantics for shapeVec, we must ask for them
				ShapedParticleContainer(ShapedParticleContainer&&) = default;
				ShapedParticleContainer& operator=(ShapedParticleContainer&&) = default;
				
				//! @brief Append a set of particles sharing the same shape
				void append(std::vector<PhatF> const& tail, 
					std::shared_ptr<ShapeFunction> const& theirSharedShape = delta);
					
				//! @brief Append a single particle
				void emplace_back(PhatF const& pHatF, 
					std::shared_ptr<ShapeFunction> const& itsShape = delta);
		};
		
		/*! \ingroup ParticleContainers
		 *  @brief A class for communicating the tracks and towers observed in a detector
		 *  
		 *  The primary purpose of this class is to take a detector's raw data 
		 *  (stored with units of energy) and boost it into a different frame,
		 *  returning the boosted observation in a format compatible with PowerSpectrum.
		*/
		class DetectorObservation : public PowerSpectrum::ParticleContainer
		{
			friend class Job_AngularResolution;
			
			public:	
				using real_t = PowerSpectrum::real_t;
				using vec3_t = PowerSpectrum::vec3_t;
				using PhatF = PowerSpectrum::PhatF;
				using VecPhatF = PowerSpectrum::VecPhatF;
				
			protected:
				virtual real_t pHatMag(size_t const i) const;
			
			public:
				VecPhatF tracks; //!< @brief Charged particles seen in the tracker
				VecPhatF towers; //!< @brief Track-subtracted neutral energy from calorimeter cells
				//! @brief The surface fraction of each tower (the tower's solid angle \f$\Omega / (4\pi) \f$)				
				std::vector<real_t> towerAreas;
				
				DetectorObservation() = default; //!< @brief Let the detector decide how to fill
				
				DetectorObservation(std::vector<vec3_t> const& tracks_in, 
					std::vector<vec3_t> const& towers_in, std::vector<real_t> const& towerAreas_in);
				
				~DetectorObservation() {} // To suppress inline warnings.
				
				virtual void NormalizeF_to(real_t const total_f);
				
				virtual real_t fTotal() const;
				virtual real_t fInner() const;		
				virtual size_t size() const;
				
				/*! @brief Check that towerAreas is properly formatted
				 * 
				 *  Either all towers have same area or each tower has a unique area.
				*/ 
				void CheckValidity_TowerArea() const;
				
				//! @brief Convert all towers to tracks and return a new DetectorObservation.
				DetectorObservation NaiveObservation() const;
				
				/*! @brief Construct a ShapedParticleContainer, 
				 *  using h_Gaussian for tracks and h_Cap for towers
				 *  
				 *  Tower's circular caps are defined from the internally stored towerAreas.
				 *  The lambda for the track's pseudo-normal distribution is determined
				 *  by solving for the lambda where a fraction \p u_track (u in [0,1])
				 *  of the track's shape lies within a circle of radius R_track 
				*/  
				PowerSpectrum::ShapedParticleContainer MakeExtensive
				(real_t const angularResolution, real_t const f_trackR = 1.,
					real_t const u_trackR = 0.9) const;
		};
		
		/*! \defgroup SmearedDistance
		 *  @brief Calculating a sample's angular distance requires
		 *  calculating the angular distance between extensive objects.
		 * 
		 *  A track (delta-distribution) striking the exact center of a tower 
		 *  (energy uniformly distributed over a circular cap of radius R)
		 *  creates an angle \f$ \xi = 0 \f$ between the two objects.
		 *  However, because the tower is extensive, 
		 *  the angle between the two packets of energy is 
		 *  \em not at zero angle because we must integrate 
		 *  over the two shape functions
		 *  	\f[ \xi_{ij}^* = \int {\rm d}\Omega \int {\rm d}\Omega^\prime
		 *    h_i(\hat{r})\,h_j(\hat{r}^\prime)\arccos(\hat{r}\cdot\hat{r}^\prime) \f]
		 *  This means that a track striking slightly off center 
		 *  from the tower is not much different than the same track
		 *  striking the exact center, because in both cases, 
		 *  much of the tower energy is distributed at a distance R/2.
		 * 
		 *  This "smeared" angle integral is difficult to calculate analytically, 
		 *  so we opt for Monte Carlo integration. 
		 *  Unfortunately, this implies a new integral for every radii (slow).
		 *  However, it we \em scale the raw angle by the tower's radius R
		 *  (or, if we're calculating the smeared angle between two towers, 
		 *  their shared radius \f$ R = \sqrt{R_1^2 + R_2^2} \f$, 
		 *  adding radii in quadrature like the variance of uncorrelated random variates)
		 *  we get an answer that is nearly independent of angular scale R.
		 *  Defining this dimensionless, raw angle \f$ r = \xi / R \f$
		 *  between the centers of the two objects (so r can go to zero), 
		 *  we calculate the dimensionless, smeared angle \f$ r^* \f$
		 *  between the packets of energy. We find that as \f$ r \to 0 \f$, 
		 *  \f$ r^* \f$ flattens to a minima. In fact, the functional 
		 *  form of this shape is well-approximated by the pseudo-hyperbola:
		 *  	\f[ r^*(r) = (a^b + r^b)^\frac{1}{b}\f]
		 *  Thus, as \f$ r \gg R \f$, \f$ r^* \sim r \f$ because 
		 *  all objects with finite extent appear point-like from far enough away.
		 *  The parameters a and b were fit to the Monte Carlo integration results for 
		 *  both track-tower angles and tower-tower angles,
		 *  and hard-coded into SmearedDistance_TrackTower() and 
		 *  SmearedDistance_TowerTower() (respectively).
		 *  Curiously, both shapes are very close to one another,
		 *  and are essentially independent of R until it approaches 30 degrees or more
		 *  (indicating towers much larger than we plan to use).
		 * 
		 *  To use the dimensionless functions, one calculates the system radius \em R, 
		 *  then finds \f$ \xi^* = R\,r^*(\xi / R) \f$.
		*/ 
		
		/*! \ingroup SmearedDistance
		 *  @brief The smeared angular distance between a delta distribution and a 
		 *  circular cap of radius R_cap, in terms of the dimensionless angle r = xi / R_cap.
		 */  
		static real_t SmearedDistance_TrackTower(real_t const r);
		
		/*! \ingroup SmearedDistance
		 *  @brief The smeared distance between two circular caps of radius 
		 *  \f$ R = \sqrt{R_1^2 + R_2^2} \f$ in terms of the dimensionless angle r = xi / R.
		 */
		static real_t SmearedDistance_TowerTower(real_t const r);
				
	private:
					GCC_IGNORE_PUSH(-Wpadded)	
		/*! \ingroup Tiles
		 *  @brief A class to transmit an linear algebra jobs to a thread in the thread pool
		 * 
		 *  Threads are given a Job, which specifies a calculation. 
		 *  Each Job is broken into tiles, whose TileSpecs are stored inside the Job.
		 *  Each Job will be dispatched to multiple threads until the job is complete. 
		 *  Then, the thread adding the final tile will release the hold on the job 
		 *  (by setting remainingTiles to zero) and notify the sub-manager 
		 *  (the thread launching the job). A possible race condition has 
		 *  the sub-manager noticing the job is done before being notified, 
		 *  deleting the Job and returning. When the thread tries to 
		 *  notify on the deleted Job ... undefined behavior.
		 *  To prevent this, we will create/access Job objects via a std::shared_ptr.
		 * 
		 *  Most of the work of Job is handled through public member functions
		 *  which hide the implementation; this protects me from myself and
		 *  simplifies the Job's user interface.
		*/ 
		class Job
		{
			public:
				using TileSpecs = TiledOuter::TileSpecs<tileWidth>;
				using TileSpecs_ptr = std::unique_ptr<TileSpecs>;
				using TileFill = TiledOuter::TileFill;
												
				typedef std::array<real_t, TileSpecs::incrementSize> array_t;
			
			protected:
				//! @brief A list of tiles which need calculation; filled by the derived ctor
				std::vector<TileSpecs_ptr> tileVec;
				
				//! @brief How many tiles are incomplete? When 0, job is done; notify sub-manager.
				//! Set by the derived ctor
				size_t remainingTiles;
				
				/*! @brief Defer initialization
				 * 
				 *  The derived ctor \em must fill tileVec and set remainingTiles.
				*/ 
				Job();
								
			private:
				/*! @brief A thread-safe iterator used to assign the next tile to each thread.
				 * 
				 * Each thread is assigned tileVec[nextTile++].
				 * We do not synchronize/wait on the dispatch of tiles, 
				 * only their completion, so a mutex-protected iterator 
				 * (for synchronization via \var done) is not necessary.
				 * Additionally, profiling (by others) indicates that 
				 * std::atomic is at least 10 times faster than a mutex-protected value.
				*/
				std::atomic<size_t> nextTile;
				
				mutable std::condition_variable done; //!< @brief Notify sub-manager of job completion.
				mutable std::mutex jobLock; //!< @brief Synchronize (done) and (remainingTiles).
				
			protected:
				//! @brief If tiles remain, return a TileSpecs pointer;
				//! return \p nullptr when no tiles remain.
				TileSpecs* NextTile();
				
				/*! @brief Finalize a threads work, given the number of tiles it completed
				 *  
				 *  Because derived Jobs do different things, this function 
				 *  does not know how to finalize job. Instead, it accepts 
				 *  the derivedJob class as an argument and uses it as a functor.
				 *  This requires accepting as many additional arguments as necessary
				 *  and perfect-forwarding them to the functor.
				 * 
				 *  \note Designing this function this way allows us to code
				 *  the thread safety in one place, with the Job specific work defined elsewhere.				 *  
				 */  
				template<typename Job_derivedClass, typename... Args>
				void Finalize_Thread(size_t const numTiles, Job_derivedClass& functor, Args&&... args)
				{
					// Don't waste time synchronizing if no tiles were pulled; nothing to do
					if(numTiles) 
					{
						assert(numTiles <= tileVec.size());
						
						//Synchronize Hl_total and remainingTiles (job completion)
						std::unique_lock<std::mutex> lock(jobLock);
						
						// Treat the derived Job class as a functor
						functor(std::forward<Args>(args)...);

						remainingTiles -= numTiles;
						assert(remainingTiles <= tileVec.size());
						
						// Unlock before notify to prevent hurry-up-and-wait
						lock.unlock();
					
						// If all tiles are done, notify the sub-manager (we are outside of lock, 
						// so this could create a race to notify, but that's not a problem).
						if(remainingTiles == 0)
						{
							// Sanity check; if the job is done, all tiles were assigned
							assert(nextTile >= tileVec.size());
							
							done.notify_one();
						}
					}
				}
				
				//! @brief Block until the job is done
				void WaitTillDone() const;
				
			public:
				/*! @brief Number of incomplete tiles
				 * 
				 *  \note This is \em not thread safe, but is only used by 
				 *  the sub-manager to determine how many threads to notify about the new job 
				 *  (notify_one or notify_all). In the case of a race condition, 
				 *  too many threads will be awoken, which likely has minimal side effects.
				 */
				size_t RemainingTiles() const {return remainingTiles;}
			
				/*!  @brief When a job is fully assigned, it can be pruned from jobQueue
				 * 
				 *  \note The result is \em not thread safe, and may soon become invalid. 
				 *  In the case of a race condition, a full assigned job will be dispatched to a thread, 
				 *  which will quickly realize that the job is full assigned and ask for another.
				 *  It will \em not attempt to do anything to the full assigned job,
				 *  so this side effect is not terrible.
				*/ 
				bool IsFullyAssigned() const {return (nextTile >= tileVec.size());}
			
				virtual void DoTiles() = 0;
		};
		
		class Job_Hl : public Job
		{
			private:
				using shapePtr_t = std::shared_ptr<ShapeFunction>;
				
				////////////////////////////////////////////////////////////
				
				using ApplyShapes_t = uint_fast8_t;
				
				/*! @brief When do we apply shape functions to \p Hl_partial in DoTiles(); 
				 *  before or after it is accumulated into a single number.
				 *
				 *  If all the rows share the same shape, and all the columns have the same shape:
				 *  \f[	H_l = h_l^{row} * h_l^{col} * <f_row| P_l( |p_row> <p_col| ) |f_col> \f]
				 *  In this case, we can multiply the shape after the accumulating Hl_accumulate 
				 *  (which uses less FLOPS). Otherwise, we must multiply shape before we accumulate.
				 *  
				 *  ApplyShapes is a bit flag; bothBefore = BitAnd(rowsBefore, colsBefore)
				 * 
				 *  \note Normally we would define an enum class inside a function
				 *  (so as not to clutter the class ... always hide as much implementation as possible).
				 *  However, we need to define bitwise operators for a strongly-typed enum.
				*/
				enum class ApplyShapes : ApplyShapes_t {after = 0, // rows and cols after
					rowsBefore = 1, // rows before, cols after
					colsBefore = 2, // cols ...
					bothBefore = 3}; // you get it
				
				// I would rather use "operator bitor", but the compiler won't take static "operator bitor"
				static ApplyShapes BitOr(ApplyShapes const left, ApplyShapes const right);				
				static ApplyShapes BitAnd(ApplyShapes const left, ApplyShapes const right);
				static bool IsBefore(ApplyShapes const val);
				
				////////////////////////////////////////////////////////////
			
				//! @brief Accumulate the partial jobs with a functor
				struct Accumulate : public std::vector<real_t>
				{
					/*! @brief Add Hl_Partial to this.
					 * 
					 *  We pas Hl_partial by reference so we can std::move it 
					 *  and steal its contents if this is empty
					*/ 
					void operator()(std::vector<real_t>& Hl_partial);
				};
			
				ShapedParticleContainer const* const rows; //!< @brief rows
				ShapedParticleContainer const* const cols; //!< @brief columns
				
				Accumulate Hl_total;
				
				size_t const lMax; //! @brief Calculate H_l form l = 1 -- lMax
				
				//! @brief Having completed \p numTiles, add \p Hl_partial to Hl_total
				void Finalize_Thread(size_t const numTiles, std::vector<real_t>& Hl_partial);
				
				/*! @brief Clone shape functions for internal use
				 * 
				 *  ShapeParticleContainer.shapeVec is a vector of pointers to shape functions.
				 *  If all the particles use the same shape function, it is the same pointer over and over.
				 *  This is beneficial because ShapeFunction.hl operates recursively, 
				 *  and remembers its last value. So the first time hl(10) is called, 
				 *  math is done, but the second time hl(10) is called, we look up the cached value.
				 *  But since ShapeFunction's are not thread-safe, we must clone any we intend to use.
				 *  Tf we have repeated pointers in shapeVec, a simple cloning 
				 *  will create a unique clone for each repeat. We therefore use 
				 *  std::map to map unique shapes to their unqiue clones.
				*/
				static std::vector<shapePtr_t> CloneShapes(std::vector<shapePtr_t> const& shapeVec, 
					size_t const begin, size_t const size);
					
			public:
				//! @brief Construct a job with a given rows and columns
				Job_Hl(size_t const lMax_in, 
					ShapedParticleContainer const& left,
					ShapedParticleContainer const& right);
				
				virtual void DoTiles() override final;
			
				/*! @brief Block until the job is complete, then return the total Hl for all tiles.
				 * 
				 *  Called by the sub-manager after dispatching the job.
				*/ 
				std::vector<real_t> Get_Hl();
		};
		
		struct AngleWeight
		{
			real_t angle; 
			real_t weight;
			
			AngleWeight() = default;
			AngleWeight(real_t const angle, real_t const weight);
			
			bool operator < (AngleWeight const& rhs) const;
			
			//! @brief Sort a vector whose head is sorted and tail in unsorted
			static void ReSort(std::vector<AngleWeight>& headTail, size_t const n_sorted);					
		};
		
		class Job_AngularResolution : public Job
		{
			private:
				enum class TileType {TrackTrack, TowerTrack, TowerTower};
			
				struct TileSpecs : public Job::TileSpecs
				{
					TileType type;
										
					TileSpecs(size_t const row_beg, size_t const num_rows,
						size_t const col_beg, size_t const num_cols, 
						TileType const type_in);
				};				
				
				//! @brief Accumulate the partial jobs with a functor
				struct Accumulate : public std::vector<AngleWeight>
				{
					/*! @brief Add Hl_Partial to this.
					 * 
					 *  We pass Hl_partial by reference so we can std::move it 
					 *  and steal its contents if this is empty.
					 * 
					 *  \warning We assume that this and angleWeight_partial are both fully sorted.
					*/ 
					void operator()(std::vector<AngleWeight>& angleWeight_partial);
				};
				
				/*! @brief Given cosXi and the tile type, calculate the extensive angle.
				 *  If this angle is smaller than xi_max, append it to angleWeight_sorted.
				 * 
				 *  Take the absolute index of the two particles
				 * 
				 *  \return The quantity of weight appended to angleWeight_sorted.
				*/ 				
				real_t NewAngle(std::vector<AngleWeight>& angleWeight_sorted,
					real_t const cosXi, real_t const weight,
					size_t const i_abs, size_t const j_abs, TileType const type) const;
				
				/*! @brief Remove the tail of angleWeight_sorted until its 
				 *  total weight is just above weight_target.
				 * 
				 *  \return The amount of weight removed from the tail.
				*/ 
				real_t StripTailWeight(std::vector<AngleWeight>& angleWeight_sorted, 
					real_t const totalWeight) const;
					
				//! @brief Having completed \p numTiles, add \p angleWeight_sorted to angleWeight_final
				void Finalize_Thread(size_t const numTiles, std::vector<AngleWeight>& angleWeight_sorted);
				
				////////////////////////////////////////////////////////////
				
				PhatFvec towers;
				PhatFvec tracks;
				
				std::vector<real_t> twrRadii;
				
				real_t weight_target;
				
				////////////////////////////////////////////////////////////
				
				/*! @brief The minimum cos(xi) for consideration as a smallest angle
				 *  (BEFORE the angle is made extensive)
				 * 
				 *  \note Use (good = (cosXi > cosXi_min))
				 * 
				 *  Set by the constructor to (angleMargin) times the expected nearest-neighbor angle.
				 *  By construction, particles cannot be more evenly distributed
				 *  than total isotropy, making min_cosXi a conservative cut
				 *  (i.e., when particles clump, more pass the cut).
				*/ 
				real_t cosXi_min;
				
				/*! @brief The maximum xi for consideration as a smallest angle
				 *  (AFTER the angle is made extensive)
				 * 
				 *  Set by the constructor to (angleMargin) times the expected nearest-neighbor angle.
				 *  By construction, particles cannot be more evenly distributed
				 *  than total isotropy, making xi_max a conservative cut
				 *  (i.e., when particles clump, more pass the cut).
				*/
				real_t xi_max;
				
				//! @brief The safety margin for cosXi_min
				static real_t constexpr angleMargin = 2.;
				
				////////////////////////////////////////////////////////////
								
				Accumulate angleWeight_final; //!< @brief The final list of smallest angles
				
			public:
				//! @brief Construct an angular resolution job for observation.
				Job_AngularResolution(DetectorObservation const& observation,
					real_t const fInner_scale, bool const xi_cut);
			
				virtual void DoTiles() override final;
			
				/*! @brief Block until the job is complete, then return the angular resolution
				 * 
				 *  Called by the sub-manager after dispatching the job.
				*/ 
				real_t Get_AngularResolution();
		};
					GCC_IGNORE_POP
		
		////////////////////////////////////////////////////////////////
		/*! \defgroup ThreadSynchronization
		 *  @brief Preventing race conditions in the thread pool requires special care.
		 * 
		 *  Two std::condition_variables allow threads to communicate 
		 *  (a thread will wait/block until it is notified).
		 *  Each needs a std::mutex (the "talking stick" of thread communication)
		 *  to synchronize the threads. This synchronization occurs though a 
		 *  "work flag" which guarantees that new action is needed.
		 *  This work flag is required because it is possible for threads to be 
		 *  awoken spuriously from a condition_variable (a limitation of implementation, not a bug), 
		 *  so threads need to be able to return to sleep immediately.
		 *  And to prevent a race condition when checking the work flag, 
		 *  it must be guarded by a mutex.
		*/ 
		
		//! \ingroup ThreadSynchronization
		//! @brief Used by Launch_Hl_Job() to notify threads that jobQueue has just grown.
		std::condition_variable newJob;
		
		//! \ingroup ThreadSynchronization
		//! @brief Used by threads to notify Hl_Job() that activeJobs has just decremented
		std::condition_variable jobDone;
		
		/*! \ingroup ThreadSynchronization
		 *  @brief Synchronizes the dispatching of Jobs to threads
		 * 
		 *  dispatch_lock synchronizes jobQueue (the work flag) and newJob (the notifier),
		 *  as well as keepAlive.
		*/ 
		std::mutex dispatch_lock;
		
		/*! \ingroup ThreadSynchronization
		 *  @brief Synchronizes the creation and completion of jobs
		 *  
		 *  threadSafe_lock synchronizes activeJobs (the work flag) and jobDone (the notifier),
		 *  as well as keepAlive.
		*/
		std::mutex threadSafe_lock;
		
		/*! \ingroup ThreadSynchronization
		 *  @brief A queue of jobs, used as first-in-first out queue (FIFO).
		 * 
		 *  The basic FIFO is std::queue, but it lacks some useful functions.
		*/ 
		std::deque<std::shared_ptr<Job>> jobQueue;
		
		/*! \ingroup ThreadSynchronization
		 *  @brief A count of active Jobs in the thread pool
		 * 
		 *  jobQueue is "popped" whenever the front Job is fully dispatched, 
		 *  \em not when that Job done. We use activeJobs to record when
		 *  a Job's final answer has returned. To ensure that a PowerSpectrum object is 
		 *  fully thread-safe, it cannot be deconstructed until activeJobs is zero.
		*/
		size_t activeJobs;
		
		// The threads are stored in the pool (a holding pen, only used during ctor and dtor)
		std::vector<std::thread> threadPool;
		
		/*! \ingroup ThreadSynchronization
		 *  @brief The thread pool is kept alive by this flag.
		 *  
		 *  keepAlive is checked in the Dispatch() loop, 
		 *  so to prevent a race condition in the Dispatch() loop during ~PowerSpectrum()
		 *  (the only place keepAlive is altered), it must be guarded by 
		 *  \em both threadSafe_lock and dispatch_lock.
		*/ 
		bool keepAlive;
		
		/*! @brief The function run by the thread pool to calculate portions of the power spectrum
		 *  
		 *  Each thread requests Jobs. When it has an active Job, 
		 *  it grabs tiles till they're all gone, then adds 
		 *  its accumulated Hl to the Job's Hl. If there are no Jobs, the thread goes idle.
		*/
		void Thread();
		
		/*! @brief Dispatch Jobs the thread pool (called by threads in Thread())
		 * 
		 *  It is extremely useful to encapsulate the following 
		 *  functionality into a single function: 
		 *  1. Dispatch Jobs to the thread pool via the return value.
		 *  2. Internally prune inactive Jobs from the jobQueue.
		 *  3. Block/wait until there are Jobs (putting threads to sleep).
		 *  4. When all jobs are complete and <tt> keepAlive == false </tt>, 
		 *     return nullptr (shared_ptr equivalent), instructing threads to exit Thread().
		 * 
		 *  \return A shared_ptr to an active Job, or nullptr when it is time for threads
		 *  to return from Thread
		*/
		std::shared_ptr<Job> Dispatch();
		
		/*! @brief Calculate the partial power spectrum for the two ensembles
		 * 
		 *  Construct a new Job (e.g. the list of tiles in its tileVec),
		 *  put the Job in the jobQueue, notify the threads, and return the result. 
		 *  This function is thread-safe; it can be called by multiple threads
		 *  simultaneously without any side effects or race conditions.
		 *  All jobs requested before the dtor is called are guaranteed to finish.
		 * 
		 *  @throws throws a runtime_error if called after the dtor.
		 */
		std::vector<real_t> Launch_Hl_Job(size_t const lMax, 
			ShapedParticleContainer const& left, ShapedParticleContainer const& right);			
			
	public:
		//! @brief Construct a thread pool of a given size.
		PowerSpectrum(size_t const numThreads = maxThreads);	
		
		//! @brief Wait for all active jobs to finish, then destroy the thread pool.
		~PowerSpectrum();
		
		/*! @brief Estimate the angular resolution of the ensemble
		 *  (as explained in "Harnessing the global correlations of the QCD power spectrum").
		 * 
		 *  Find all inter-particle angles using the \ref SmearedDistance
		 *  (treating tracks as delta distributions and towers as circular caps).
		 *  Sort the angles from smallest to largest, 
		 *  each with weight \f$ w_k \equiv w_{ij} = f_i\,f_j \f$, 
		 *  and find the set of \em M smallest angles whose 
		 *  total weight exceeds the asymptotic plateau by some factor \p ff_fraction:
		 *  \f[ \sum_{k=1}^M w_k \ge {\tt ff\_fraction}\times \langle f | f \rangle \f]
		 *  Return the geometric mean of this set of \em M smallest angles.
		*/ 
		real_t AngularResolution(DetectorObservation const& observation, 
			real_t const fInner_scale = real_t(1));
			
		/*! @brief Calculates and sorts \em every inter-particle angle to find the angular resolution
		 * 
		 *  This function uses a single thread \em and a ton of memory,
		 *  and is provided primarily to test the parallel version (\ref AngularResolution) 
		*/ 
		static real_t AngularResolution_Slow(DetectorObservation const& observation, 
			real_t const fInner_scale = real_t(1));
	
		/*! @brief Given an ensemble of particles, calculate their power spectrum 
		 *  from \f$ \ell = 1 \f$ to \p lMax
		*/ 
		std::vector<real_t> Hl_Obs(size_t const lMax, 
			ShapedParticleContainer const& particles);
		
		/*! @brief Given an ensemble of jets, calculate their power spectrum 
		 *  from \f$ \ell = 1 \f$ to \p lMax
		 * 
		 *  The jet model may need to account for the attenuating "filter" of the
		 *  detector shape functions (especially the towers). This can be applied as:
		 *  \f[ H_\ell = H_\ell^{\rm orig} \times h_{\ell, {\rm filter}}^2\f]
		 * 
		 *  \param hl_detector_Filter 
		 *  The "up" coefficient (\f$ \ell > 0 \f$) for the detector apparatus.
		 *  If an empty vector is supplied, the power spectrum is returned unaltered.
		 *   
		 *  \throws Throws \c runtime_error if \p hl_detectorFilter is shorter than \p lMax
		*/ 
		std::vector<real_t> Hl_Jet(size_t const lMax,
			ShapedParticleContainer const& jets, 
			std::vector<real_t> const& hl_detector_Filter = std::vector<real_t>());
		
		/*! @brief Calculate the hybrid power spectrum from \f$ \ell = 1 \f$ to \p lMax
		 * 
		 *  The hybrid power spectrum combines the jet model and detector 
		 *  observation into a hybrid event shape 
		 *  	\f[ \rho_{\rm hybrid} = \frac{1}{2}(\rho_{\rm jet} + \rho_{\rm obs}) \f]
		 *  This produces a hybrid power spectrum
		 *  	\f[ H_\ell^{\rm hybrid} = \frac{1}{4}
		 *    (H_\ell^{\rm jet} + 2 H_\ell^{\rm jet,hybrid} + H_\ell^{\rm obs})\f]
		 * 
		 *  The jet model may need to account for the attenuating "filter" of the
		 *  detector shape functions (especially the extensive towers). This can be applied as:
		 *  \f[ H_\ell^{\rm jet} = H_\ell^{\rm jet,orig} \times h_{\ell, {\rm filter}}^2\f]
		 *  \f[ H_\ell^{\rm jet, particle} = H_\ell^{\rm jet,particle,orig} \times h_{\ell, {\rm filter}}\f]
		 * 
		 *  \param hl_detector_Filter 
		 *  The "up" coefficient (\f$ \ell > 0 \f$) for the detector apparatus.
		 *  If an empty vector is supplied, the power spectrum is returned unaltered.*  
		 * 
		 *  \param Hl_Obs_in 
		 *  A pre-computed power spectrum for the particles
		 * 
		 *  \throws Throws \c runtime_error if \p hl_detectorFilter is shorter than \p lMax
		*/ 
		std::vector<real_t> Hl_Hybrid(size_t const lMax,
			ShapedParticleContainer const& jets, 
			std::vector<real_t> const& hl_detector_Filter,
			ShapedParticleContainer const& particles,
			std::vector<real_t> const& Hl_Obs_in = std::vector<real_t>());
		
		/*! @brief Write a set of power spectra to a human-readable file
		 * 
		 *  The file format is designed to be read by gnuplot.
		 *  Each power spectrum gets its own column, with the first column being \f$ \ell \f$
		 *  We \em assume that each power spectrum starts with \f$ \ell = 1 \f$.
		 *  Any power spectrum which is shorter than the longest set 
		 *  will be padded with -1. (a nonsense value for power spectra).  
		 * 
		 *  \note To write a single power spectrum, wrap it in curly brackets
		 *  to create an initializer list; the compiler should know what to do.
		 * 
		 *  \param header
		 *  The first line in the file, to identify the columns
		 *  (automatically pre-pended with gnuplot comment character '#')
		*/ 
		static void WriteToFile(std::string const& filePath, 
			std::vector<std::vector<real_t>> const& Hl_set,
			std::string const& header = "");
		
		/*! @brief Take a set of power spectra and calculate their angular correlation functions A(z)
		 *  
		 *  The angular correlation function (alternatively known as the 
		 *  Energy-Energy correlation or the auto-correlation)
		 *  uses the power spectrum as the coefficients in a Legendre series:
		 *  	\f[ A(z) = \sum_{\ell = 0}^{\ell_{\rm max}} (2\ell+1) H_\ell\, P_\ell(z) \f]
		 *  For a power spectrum which asymptotically vanishes, this series converges.
		 *  In this implementation, \f$ \ell_{\rm max} \f$ is determined from the 
		 *  length of the individual power spectra.
		 * 
		 *  \param zSamples
		 *  The number of z-values sampled (uniformly) between -1 and 1 (exclusive)
		 *  
		 *  \return Returns a std::pair, with \c first being the z-values and 
		 *  \c second being A(z) sampled at those values.
		*/  
		static std::pair<std::vector<real_t>, std::vector<std::vector<real_t>>>
			AngularCorrelation(std::vector<std::vector<real_t>> const& Hl_set, 
				size_t const zSamples = 2048);
		
		/*! @brief Take a set of power spectra and calculate their angular correlation functions A(z),
		 *  and write them to a human-readable file.
		 *  
		 *  The file format is designed to be read by gnuplot.
		 *  Each A(z) gets its own column, with the first column being the z-values.
		 *  See AngularCorrelation for an explanation of A(z).
		 * 
		 *  \param zSamples
		 *  The number of z-values sampled (uniformly) between -1 and 1 (exclusive)
		 * 
		 *  \param header
		 *  The first line in the file, to identify the columns
		 *  (automatically pre-pended with gnuplot comment character '#')
		*/ 
		static void Write_AngularCorrelation(std::string const& filePath, 
			std::vector<std::vector<real_t>> const& Hl_set, size_t const zSamples = 2048, 
			std::string const& header = "");
};

GCC_IGNORE_POP

#endif
