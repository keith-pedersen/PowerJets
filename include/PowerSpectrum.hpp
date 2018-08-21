#ifndef POWER_SPECTRUM
#define POWER_SPECTRUM

// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PowerJets.hpp"
#include "kdp/kdpTools.hpp"
#include "Pythia8/Event.h"
#include "NjetModel.hpp"
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
		 *  Why? Let me briefly explain. The problem with ...
		 * 
		 *  \verbatim 
		 for(i = 0; i < size: ++i)
		    for(j = 0; j < size; ++j)
		       doMath(lhs[i], rhs[j])
		 \endverbatim
		 * 
		 *  is that we iterate over the whole row before moving to the next row.
		 *  Hence, the inner loop always iterates over every column 
		 *  (a large swath of memory which must be repeatedly requested, 
		 *  and is perhaps too large to sit in the CPU's cache). 
		 *  splits the matrix into smaller squares, each with a local origin (iTile, jTile):
		 * 
		 *  \verbatim
		 for(i = 0; i < 16: ++i)
		    for(j = 0; j < 16; ++j)
		       doMath(lhs[iTile + i], rhs[jTile + j])
		 \endverbatim
		 *  This allows the calculation to repeatedly access the same 16 rows and columns, 
		 *  which fit into a few lines of the CPU cache. Thus, repeatedly accessing these 
		 *  32 elements should often hit the CPU cache, instead of having to wait for main memory.
		 *  This is very cache efficient, so that the calculation becomes
		 *  more limited by the FLOPS, and less by the memory bus.
		 *  Similarly, by choosing a tileWidth = \f$ 2^k \f$, memory should be aligned.
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
			
				//! @brief Create an ensemble of extensive jets, auto-normalized to their total energy
				ShapedParticleContainer(std::vector<ShapedJet> const& jets);
				
				/*! @brief Emplace particles all sharing the same shape, 
				 *  with auto-normalization to their total energy.
				*/ 
				ShapedParticleContainer(std::vector<vec3_t> const& particles, 
					std::shared_ptr<ShapeFunction> const theirSharedShape = delta);
				
				/*! @brief Emplace particles all sharing the same shape; 
				 *  no normalization (verbatim copy).
				 */
				ShapedParticleContainer(std::vector<PhatF> const& particles, 
					std::shared_ptr<ShapeFunction> const theirSharedShape = delta);
					
				// Note; to get move semantics for shapeVec, we must ask for them
				ShapedParticleContainer(ShapedParticleContainer&&) = default;
				ShapedParticleContainer& operator=(ShapedParticleContainer&&) = default;
				
				//! @brief Append a set of particles sharing the same shape
				void append(std::vector<PhatF> const& tail, 
					std::shared_ptr<ShapeFunction> const theirSharedShape = delta);
					
				//! @brief Append a single particle
				void emplace_back(PhatF const& pHatF, 
					std::shared_ptr<ShapeFunction> const itsShape = delta);
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
				 *  from the samples angular resolution \f$ \xi_{\min} \f$.
				 * 
				 *  We solve for the lambda where a fraction \p u_trackR (u in [0,1])
				 *  of the track's shape lies within a circle of radius R = \p f_trackR * \f$ \xi_{\min} \f$. 
				*/  
				PowerSpectrum::ShapedParticleContainer MakeExtensive(double const f_trackR = 1.,
					double const u_trackR = 0.9) const;
				
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
				real_t AngularResolution(real_t const ff_fraction = real_t(1)) const;
				
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
		};
				
	private:
		/*! \ingroup Tiles
		 *  @brief The type of tile for a tiled outer product
		 *  
		 *  When we split the outer product into tiles, there are 
		 *  five different types of tile we can have:
		 *  (D)iagonal, (C)enter, (B)ottom, (R)ight, and (F)inal.
		 *  Each will require different treatment inside Hl_Thread(), 
		 *  depending on if the outer product is symmetric.
		 *  \verbatim
		    u x u (symmetric)     u x v (asymmetric)
		    +-----+-----+---     +-----------------+---
		    | D D | * * | *      | C C | C C | C C | R
		    | D D | * * | *      | C C | C C | C C | R
		    +-----+-----+---     +-----+-----+-----+---
		    | C C | D D | *      | B B | B B | B B | F
		    | C C | D D | * 
		    +-----+-----+---
		    | B B | B B | F 
		 \endverbatim 
		 *  SYMMETRIC: We do not need to calculate the redundant upper tiles, 
		 *  we can simply double off-diagonal tiles (i.e. RIGHT are the same as BOTTOM, 
		 *  and each CENTER show up twice, so simply double their contribution).
		 *  However, DIAGONAL tiles are kept full (and thus half-redundant)
		 *  because it is faster than jagged, non-redundant tiles. 
		 *  This requires \em not doubling the DIAGONAL tiles' contribution.
		 *  BOTTOM tiles have less rows than a full tile, but every row is a full tileWidth;
		 *  this keeps them efficient for auto-vectorized SIMD instructions.
		 *  The FINAL tile is jagged (diagonal and lower elements only)
		 *  \em only if it is small; otherwise it is faster to make FINAL a 
		 *  redundant square (like DIAGONAL, albeit with rows that are not full).
		 * 
		 *  ASYMMETRIC: There is no redundancy; all CENTER tiles are unique.
		 *  BOTTOM tiles have less rows than a full tile, but ever row is a full tileWidth.
		 *  Conversely, RIGHT tiles have lass columns than a full tile, 
		 *  but every column is a full tileWidth. Since the outer product loops 
		 *  iterate over columns in the inner loop (which is always the one that is vectorized), 
		 *  RIGHT tiles are less efficient. To fix this problem,
		 *  we treat the columns like rows for RIGHT tiles
		 *  (i.e. we swap the pointers to u and v and also iTile and jTile).
		 *  The FINAL tile is square, but its rows are not full.
		*/ 
		enum class TileType {DIAGONAL, CENTRAL, BOTTOM, RIGHT, FINAL};
	
			GCC_IGNORE_PUSH(-Wpadded)
		//! \ingroup Tiles
		//! @brief A tile is specified by its boundaries and type
		struct TileSpecs
		{
			size_t row_beg; //!< @brief The index of the first row
			size_t row_width; //!< @brief The number of rows
			size_t col_beg; //!< @brief The index of the first column
			size_t col_width; //!< @brief The number of columns
			TileType type;
		};
		
		/*! \ingroup Tiles
		 *  @brief A class to transmit an Hl calculation to a thread in the thread pool
		 * 
		 *  Threads are given a Job, which specifies an H_l calculation. 
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
			private: 
				//! @brief A list of tiles which need calculation
				std::vector<TileSpecs> tileVec;
				
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
				
				std::condition_variable done; //!< @brief Notify sub-manager of job completion.
				std::mutex jobLock; //!< @brief Synchronize (done) and (remainingTiles).
				//! @brief How many tiles are incomplete? When 0, job is done; notify sub-manager.
				size_t remainingTiles;
				
				std::vector<real_t> Hl_total; //!< @brief The total Hl accumulated by threads running the job
				
			public:				
				/*! @defgroup LeftRight
				 *  
				 *  By default, the "left" supplies the rows and the "right" the columns.
				 *  However, this is reversed for RIGHT tiles (right is rows, left is columns) 
				 *  so that only FINAL tiles have half-full rows.
				 *  This scheme improves the efficiency of SIMD vectorization.
				*/
				
				ShapedParticleContainer const* left; //!< @brief rows @ingroup LeftRight 
				ShapedParticleContainer const* right; //!< @brief columns @ingroup LeftRight 
				
				size_t lMax; //! @brief Calculate H_l form l = 1 -- lMax
				
				/*! @brief Construct a job with a given left and right containers, 
				 *  a given lMax, and stealing/moving the vector of TileSpecs
				 *  (which was filled especially for this object)
				*/ 
				Job(ShapedParticleContainer const* const left_in,
					ShapedParticleContainer const* const right_in,
					size_t const lMax_in, 
					std::vector<TileSpecs>&& tileVec_in);
			
				//! @brief If tiles remain, fill the next TileSpec into \p tile.
				//! Return false when no tiles remain (signifying that \p tile was not altered).
				bool GetTile(TileSpecs& tile);
				
				/*! @brief Number of incomplete tiles
				 * 
				 *  \warning This is \em not thread safe, but is only used by 
				 *  sub-manager to determine how many threads to notify about the new job 
				 *  (notify_one or notify_all). In the case of a race condition, 
				 *  too many threads will be awoken, which likely has minimal side effects.
				 */
				size_t RemainingTiles() const {return remainingTiles;}
			
				/*!  @brief When a job is fully assigned, it can be pruned from jobQueue
				 * 
				 *  \note The result is \em not thread safe, and may soon become invalid. 
				 *  In the case of a race condition, a finished job will be dispatched to a thread, 
				 *  which will quickly realize that the job is done and ask for another.
				 *  It will \em not attempt to do anything to the finished job,
				 *  so this side effect is not terrible.
				*/ 
				bool IsFullyAssigned() const {return nextTile >= tileVec.size();}
				
				/*! @brief Take the \p Hl_partial calculated by one thread 
				 *  (for all the tiles it was assigned in this job) and add it to Hl_total.
				 * 
				 *  We need to know how many tiles this \p Hl_partial is for (i.e., \p numTiles)
				 *  so we can decrement \c remainingTiles (guarded by jobLock).
				*/ 
				void Add_Hl(std::vector<real_t>& Hl_partial, size_t const numTiles);
				
				/*! @brief Block until the job is complete, then return the total Hl for all tiles.
				 * 
				 *  Called by the sub-manager after dispatching the job.
				*/ 
				std::vector<real_t> Get_Hl();
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
		//! @brief Used by Hl_Job() to notify threads that jobQueue has just grown.
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
		void Hl_Thread();
		
		/*! @brief Dispatch Jobs the thread pool (called by threads in Hl_Thread())
		 * 
		 *  It is extremely useful to encapsulate the following 
		 *  functionality into a single function: 
		 *  1. Dispatch Jobs to the thread pool via the return value.
		 *  2. Internally prune inactive Jobs from the jobQueue.
		 *  3. Block/wait until there are Jobs (putting threads to sleep).
		 *  4. When all jobs are complete and <tt> keepAlive == false </tt>, 
		 *     return nullptr (shared_ptr equivalent), instructing threads to exit Hl_Thread().
		 * 
		 *  \return A shared_ptr to an active Job, or nullptr when it is time for threads
		 *  to return from Hl_Thread
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
		std::vector<real_t> Hl_Job(ShapedParticleContainer const* const left_in, 
			ShapedParticleContainer const* const right_in, size_t const lMax);
	
	public:
		//! @brief Construct a thread pool of a given size.
		PowerSpectrum(size_t const numThreads = maxThreads);	
		
		//! @brief Wait for all active jobs to finish, then destroy the thread pool.
		~PowerSpectrum();
	
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
			std::vector<ShapedJet> const& jets, 
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
