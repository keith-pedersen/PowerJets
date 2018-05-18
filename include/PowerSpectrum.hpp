#ifndef POWER_SPECTRUM
#define POWER_SPECTRUM

#include "PowerJets.hpp"
#include "kdp/kdpTools.hpp"
//~ #include "SpectralPower.hpp"
#include "Pythia8/Event.h"
#include "NjetModel.hpp"
#include <memory> // shared_ptr
#include <atomic>
#include <condition_variable>
#include <thread>

/*! @brief A class for parallel, cache-efficient calculation of power spectrum H_l
 * 
 *  This objects manages a pool of threads which are awoken to calculate H_l, 
 *  then put back to sleep when the calculation is done. 
 *  This is nearly twice as efficient as the previous iteration, 
 *  which created and destroyed its threads for each job.
*/
class PowerSpectrum
{
	public: 
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
				
		// In the future, these can be dynamic quantities, 
		// but we don't need that now so we'll let it be
		static constexpr size_t maxThreads = 4;
		static constexpr size_t minTilesPerThread = 2;
		
		// This should be hard-coded for compiling
		// 16 is the optimal number; not too big, not too small.
		// 8 and 64 are each 40% slower. 
		// 16 is two cache lines on my Intel i7. 
		static constexpr size_t tileWidth = 16;
		
		//! @brief A simple struct for storing particle information
		//  repeatedly used by the Hl calculation
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
			
			//! @brief Construct a vector of PhatF, normalizing f for the collection
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
			// Keep x,y,z,f hidden from everyone but the 
			// (and any of its nested classes, which have the same friend rights as SpectralPower).
			friend class PowerSpectrum;
			friend class SpectralPower;			
			
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
				// But to get move semantics (instead of copy), we have to ask for them
				PhatFvec(PhatFvec&&) = default;
				PhatFvec& operator=(PhatFvec&&) = default;
				
				inline size_t size() const {return f.size();}
				void reserve(size_t const reserveSize);
				void clear();
				
				void emplace_back(PhatF const& pHatF);
				
				// Less versatile than insert, but there's no need to define iterators
				void append(PhatFvec const& tail);
				
				//! @brief Join two PhatFvec returned by separate function calls
				static PhatFvec Join(PhatFvec&& first, PhatFvec&& second);
		};
		
		/*! @brief An extension of PhatFvec, but with the addition of particle shape. 
		 *  This is the most general tool for calculating spectral power
		 *  (assuming azimuthally symmetric shape functions).
		 */ 
		class ShapedParticleContainer : public PhatFvec
		{
			friend class PowerSpectrum;
			
			private:
				/* There are three cases for shapes in the container: 
				 * 1. Every particle has the same shape (e.g. tracks)
				 * 2. Sets of adjacent particles have the same shape (e.g. towers in a 
				 * calorimeter whose bands do not have identical solid angle, as in a hadronic calorimeter)
				 * 3. Every particle has unique shape (e.g. jets).
				 * 
				 * Because ShapeFunction objects are not thread safe, 
				 * they will always be cloned before being used in Hl_Thread. 
				 * Therefore, the easiest way to accommodate all three scenarios
				 * is simply to store pointers to the supplied ShapeFunctions, 
				 * one for each particle (repeated shapes having repeated pointers).
				 * However, this has the nasty side effect that it assumes that 
				 * the externally supplied ShapeFunction will be kept alive
				 * and/or not move. A safer bet is to clone all incoming ShapeFunctions.
				 * To simplify garbage collection (e.g. in case of copy), 
				 * we use shared_ptr.
				*/
				std::vector<std::shared_ptr<ShapeFunction>> shapeVec;
				
			public:
				ShapedParticleContainer() {}
			
				ShapedParticleContainer(std::vector<ShapedJet> const& jets);
				ShapedParticleContainer(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape);
				//~ ~ShapedParticleContainer();
				
				// Warning; to get move semantics for shapeStore, 
				// we must explicitly invoke move ctor and assignment
				ShapedParticleContainer(ShapedParticleContainer&&) = default;
				ShapedParticleContainer& operator=(ShapedParticleContainer&&) = default;
				
				void append(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape);
		};
				
	private:
		/*! @brief The type of tile for tiled computation
		 * 
		 *  When we split the outer product into tiles, there are 
		 *  five different types of tile we can have. 
		 *  Each will require different treatment. 
		 * 
		 *  Note that for symmetric outer products, 
		 *  we do not need to calculate the redundant upper tiles.
		 *  For example, the RIGHT are the same as the BOTTOM, 
		 *  so we can account for them by doubling the BOTTOM contribution.
		 * 
		 *   u x u (symmetric)     u x v (asymmetric)
		 *  +-----+-----+---     +-----------------+---
		 *  | D D | # # | #      | C C | C C | C C | R
		 *  | D D | # # | #      | C C | C C | C C | R
		 *  +-----+-----+---     +-----+-----+-----+---
		 *  | C C | D D | #      | B B | B B | B B | F
		 *  | C C | D D | # 
		 *  +-----+-----+---
		 *  | B B | B B | F 
		*/ 
		enum class TileType {DIAGONAL, CENTRAL, BOTTOM, RIGHT, FINAL};
	
			GCC_IGNORE(-Wpadded)
		//! @brief A tile is specified by its boundaries and type
		struct TileSpecs
		{
			size_t row_beg; //! @brief The index of the first row
			size_t row_width; //! @brief The number of rows
			size_t col_beg; //! @brief The index of the first column
			size_t col_width; //! @brief The number of columns
			TileType type;
		};
			GCC_IGNORE_END
		
		// By default, the "left" supplies the rows and the "right" the columns.
		// However, this is reverse for RIGHT tiles, so that only the FINAL 
		// tiles have half-full rows.
		ShapedParticleContainer const* left;
		ShapedParticleContainer const* right;
		
		size_t lMax_internal; //! @brief Calculate H_l form l = 1 -- lMax_internal
		
		// A list of tiles which need calculation
		std::vector<TileSpecs> tileVec;
		// A thread safe iterator used to assign the next tile to each thread.
		// Profiling (by others) indicates that std::atomic is at least 
		// O(10) times faster than using a mutex
		std::atomic<size_t> nextTile;
		
		std::vector<real_t> Hl_total; // The total Hl
		
		////////////////////////////////////////////////////////////////
		// Several variables are necessary for managing the thread pool.
		
		// The number of idle threads waiting for orders.
		// This is used to monitor job completion (done when all threads are idle).
		std::atomic<size_t> idle;
		
		// Two condition variables (wait/notify) are used: 
		// Threads wait for newJob, which is notified by the manager (this object).
		// The manager waits for jobDone, and is notified when threads go idle.
		std::condition_variable newJob, jobDone;
		
		// Communication between the threads and the manager is controlled by syncLock
		std::mutex syncLock; // The "talking stick" of manager-thread communication
		
		// It is simple to ensure that this object itself is thread-safe; 
		// this mutex ensures that only one job can be dispatched at a time, 
		// and that the job must be finished before the next job is dispatched. 
		std::mutex threadSafe;
		
		// The threads are stored in the pool
		std::vector<std::thread> threadPool;
		
		// The thread pool is kept alive by this boolean; the dtor sets it to false
		bool keepAlive;
		
		//! @brief Each thread grabs tiles till they're all gone, then 
		//  adds the sum of their Hl to Hl_total. Then it does idle until there's a new job.
		void Hl_Thread();
		
		//! @brief Verify a valid wakeup of threads by the managers; 
		//  either there is new work to do, or it is time to go peacefully into the night.
		bool ValidWakeup() {return (nextTile < tileVec.size()) or (not keepAlive);}	
		
		/*! @brief Construct the list of tiles in the outer product, 
		 *  launch the threads, collect the result and return, 
		 *  putting all threads back to sleep.
		 * 
		 *  This function is thread-safe; it can only be called by one thread at a time, 
		 *  which prevents two simultaneous jobs from thrashing each other.
		 */
		std::vector<real_t> Hl(ShapedParticleContainer const* const left_in, 
			ShapedParticleContainer const* const right_in, size_t const lMax);
	
	public:
		//! @brief Construct a thread management object with the given pool size
		PowerSpectrum(size_t const numThreads = 4);	
		
		//! @brief Gracefully disband the thread pool
		~PowerSpectrum();
	
		//! @brief Calculate the power spectrum for a set of particles
		std::vector<real_t> Hl_Obs(size_t const lMax, 
			ShapedParticleContainer const& particles);
		
		/*! @brief Calculate the power spectrum for a set of jets
		 * 
		 *  hl_onAxis_Filter applies the filter of the detector elements.
		*/ 
		std::vector<real_t> Hl_Jet(size_t const lMax,
			ShapedParticleContainer const& jets, std::vector<real_t> const& hl_onAxis_Filter);
			
		//! @brief Calculate the power spectrum for a set of jets
		std::vector<real_t> Hl_Jet(size_t const lMax, 
			std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter);
		
		//! @brief Calculate the hybrid power spectrum for rho = 0.5*(jets + particles)
		std::vector<real_t> Hl_Hybrid(size_t const lMax,
			std::vector<ShapedJet> const& jets_in, std::vector<real_t> const& hl_onAxis_Filter,
			ShapedParticleContainer const& particles,
			std::vector<real_t> const& Hl_Obs_in = std::vector<real_t>());
};

#endif
