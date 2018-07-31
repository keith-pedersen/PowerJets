#ifndef POWER_SPECTRUM
#define POWER_SPECTRUM

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

/*! @brief A class managing parallel, vectorized, cache-efficient calculation of a power spectrum H_l
 * 
 *  This object is thread safe; it can be called my multiple threads at a time.
 *  Deconstruction will block until all active jobs have finished and returned.
 * 
 *  This class is a useful template for the following concepts in 
 *  high-performance computing:
 *  - multi-threading (thread safety via mutex).
 *  - inter-thread communication (wait and notify via condition_variable).
 *  - a thread pool (at least twice as efficient as creating new threads for every job).
 *  - tiling (make linear algebra cache-efficient by breaking the job into squares).
 *  - auto-vectorization (SIMD instructions via specially designed for loops and data structures).
 *      - e.g. the object-of-vectors paradigm (versus the vector-of-objects).
 *  - binary accumulation (take CARE of cancellation and rounding error in large sums).
 *  - hidden implementation (private data structures can change without altering the API).
 *  - move semantics (request auto-generated move semantics when class members have them).
*/
class PowerSpectrum
{
	public: 
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
				
		// In the future, these can be dynamic quantities, 
		// but we don't need that now so we'll hard-code it.
		static constexpr size_t maxThreads = 4;
		static constexpr size_t minTilesPerThread = 1;
		
		// This should be a power-of-2, and hard-coded for compiler optimization.
		// 16 is a good number; not too big, not too small.
		// 8 and 64 are each 40% slower. 
		// 16 is two cache lines on my Intel i7, and the optimal power of two.
		static constexpr size_t tileWidth = 16;
		
		//////////////////////////////////////////////////////////////////
		// First we define a few nested classes used for
		// interfacing with the class methods.
		
		//! @brief A simple struct for storing the primary particle information
		//  used during the Hl calculation
		struct PhatF
		{
			vec3_t pHat; //!< @brief Unit-direction of travel.
			real_t f; //!< @brief Energy fraction (relative to total detected energy).
			
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
			static std::vector<PhatF> To_PhatF_Vec(std::vector<T> const& originalVec, 
				real_t const totalE_in = real_t(-1))
			{
				std::vector<PhatF> convertedVec;
				
				real_t totalE = real_t(0);
				for(auto const& original : originalVec)
				{
					convertedVec.emplace_back(original);
					totalE += convertedVec.back().f;
				}		
								
				if(totalE_in > real_t(0))
				{
					assert(totalE <= totalE_in);
					totalE = totalE_in;
				}					
				
				for(auto& converted : convertedVec)
					converted.f /= totalE;
					
				return convertedVec;
			}
			
			static real_t fInner(std::vector<PhatF> const& particles);
		};

		/*! @brief A collection of PhatF (an object-of-vectors)
		 * 
		 *  When we take the outer products \f$ \hat{p}_i \cdot \hat{p}_j \f$
		 *  and \f$ f_i f_j \f$, the object-of-vectors paradigm is much faster than the 
		 *  vector-of-objects paradigm (std::vector<PhatF>), because the 
		 *  object-of-vectors allows SIMD vectorization of the vector dot-product
		 *  (e.g. loop over i to calculate dot[k] += vec_x[i] * vec_x[j])
		 *  @note We deliberately emulate many of the functions of std::vector
		*/ 
		class PhatFvec
		{
			// Keep x,y,z,f hidden from everyone but PowerSpectrum
			// (and any of its nested classes, which have the same friend rights).
			friend class PowerSpectrum;
			friend class SpectralPower;
			
			private:
				std::vector<real_t> x, y, z;
				std::vector<real_t> f;
			
			public:
				PhatFvec() {} //!< @brief Construct an empty PhatFvec.
				~PhatFvec() {}
				
				/*! @brief Convert a std::vector<PhatF> into a PhatVec
				 *  (vector-of-classes to class-of-vectors); a verbatim copy.
				 * 
				 *  @param orig 	the std::vector<PhatF>
				 *  @param normalize the pHat in the vector-of-classes
				*/ 
				PhatFvec(std::vector<PhatF> const& orig);
								
				// We can use implicit copy and move for ctors and assigment, 
				// because x, y, z, and f all have them defined.
				// To get move semantics (instead of copy), we have to ask for them.
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
				ShapedParticleContainer(std::vector<PhatF> const& particles, 
					ShapeFunction const& theirSharedShape = h_Delta());
				ShapedParticleContainer(std::vector<vec3_t> const& particles, 
					ShapeFunction const& theirSharedShape = h_Delta());
				//~ ~ShapedParticleContainer();
				
				// Warning; to get move semantics for shapeStore, 
				// we must explicitly invoke move ctor and assignment
				ShapedParticleContainer(ShapedParticleContainer&&) = default;
				ShapedParticleContainer& operator=(ShapedParticleContainer&&) = default;
				
				void append(std::vector<PhatF> const& particles, 
					ShapeFunction const& theirSharedShape = h_Delta());
		};
				
	private:
		//////////////////////////////////////////////////////////////////
		/* TILING: obs will be specified and dispatched via a number of tiles.
		 * The problem with ...
		 * 
		 * for(i = 0; i < size: ++i)
		 * 	for(j = 0; j < size; ++j)
		 * 		doMath[i][j]
		 * 
		 * is that we iterate over the whole row before moving to the next row. 
		 * Hence, the inner loop always iterates over every column 
		 * (a large swath of memory). Tiling splits the matrix into smaller squares
		 * with an origin (iTile, jTile):
		 * 
		 * for(i = 0; i < 16: ++i)
		 * 	for(j = 0; j < 16; ++j)
		 * 		doMath[iTile + i][jTile + j]
		 * 
		 * This allows the tile to repeatedly access the same 16 rows and columns, 
		 * which fit into a few lines in the CPU cache. This is very cache efficient, 
		 * so that the calculation is limited by the FLOPS, not the memory bus.
		 * Similarly, by choosing a tileWidth = power-of-2, memory will be aligned, 
		 * and a small tileWidth allows easier vectorization of the inner loop.
		 */		  
	
		/*! @brief The type of tile for tiled computation
		 * 
		 *  When we split the outer product into tiles, there are 
		 *  five different types of tile we can have. 
		 *  Each will require different treatment inside Hl_Thread().
		 * 
		 *  For symmetric outer products, we do not need to calculate 
		 *  the redundant upper tiles, we can simply double off-diagonal 
		 *  (i.e. the RIGHT are the same as the BOTTOM, 
		 *  and the CENTER show up twice, so simply double their contribution)
		 * 
		 *   u x u (symmetric)     u x v (asymmetric)
		 *  +-----+-----+---     +-----------------+---
		 *  | D D | * * | *      | C C | C C | C C | R
		 *  | D D | * * | *      | C C | C C | C C | R
		 *  +-----+-----+---     +-----+-----+-----+---
		 *  | C C | D D | *      | B B | B B | B B | F
		 *  | C C | D D | * 
		 *  +-----+-----+---
		 *  | B B | B B | F 
		 * 
		 *  For symmetric, DIAGONAL tiles are treated differently than CENTER because
		 *  it is faster to make them full (and thus redundant), 
		 *  than jagged (and non-redundant). This requires 
		 *  *not* doubling the DIAGONAL tiles' contribution.
		*/ 
		enum class TileType {DIAGONAL, CENTRAL, BOTTOM, RIGHT, FINAL};
	
			GCC_IGNORE_PUSH(-Wpadded)
		//! @brief A tile is specified by its boundaries and type
		struct TileSpecs
		{
			size_t row_beg; //! @brief The index of the first row
			size_t row_width; //! @brief The number of rows
			size_t col_beg; //! @brief The index of the first column
			size_t col_width; //! @brief The number of columns
			TileType type;
		};
		
		/* Threads are given a Job, which specifies the calculation of an H_l. 
		 * Each Job usually has many tiles, which are stored inside the Job.
		 * Each Job will be dispatched to multiple threads (as a pointer) 
		 * until the job is complete. Then, the thread adding the final piece
		 * will release the hold on the job and notify the sub-manager (the thread launching the job). 
		 * A possible race condition has the sub-manager noticing the job is done
		 * before being notified, deleting the job and returning. 
		 * When the thread tries to notify on the deleted job ... undefined behavior.
		 * To prevent this, we will create/access Job objects via a shared_ptr.
		 * 
		 * Most of the work of Job is handled through public member functions
		 * which hide the implementation; this protects me from myself, 
		 * and also speeds up operation because Hl_Thread is not 
		 * constantly dereferencing the object pointer, then dereferencing its internal elements.
		*/ 
		class Job
		{
			private: 
				// A list of tiles which need calculation
				std::vector<TileSpecs> tileVec;
				
				// A thread-safe iterator used to assign the next tile to each thread.
				// We do not synchronize/wait on the dispatch of tiles, 
				// only their completion, so a mutex-protected iterator is not necessary.
				// Profiling (by others) indicates that std::atomic is at least 
				// 10 times faster than a mutex-protected value.
				std::atomic<size_t> nextTile;
				
				std::condition_variable done; // Notify sub-manager of job completion.
				std::mutex jobLock; // Synchronize (done) and (remainingTiles).
				size_t remainingTiles; // How many tiles remain? When 0, job is done; notify.
				
				std::vector<real_t> Hl_total; // The total Hl accumulated by threads running the job
				
			public:				
				// By default, the "left" supplies the rows and the "right" the columns.
				// However, this is reversed for RIGHT tiles, so that only the FINAL 
				// tiles have half-full rows.
				ShapedParticleContainer const* left;
				ShapedParticleContainer const* right;
				
				size_t lMax; //! @brief Calculate H_l form l = 1 -- lMax
				
				// Construct a job by stealing the vector of tiles (filled for this object)
				Job(ShapedParticleContainer const* const left_in,
					ShapedParticleContainer const* const right_in,
					size_t const lMax_in, 
					std::vector<TileSpecs>&& tileVec_in);
			
				//! @brief If tiles remain, fill the next TileSpecs into the argument.
				//  Return false when no tiles remain and tile was not altered.
				bool GetTile(TileSpecs& tile);
				
				size_t RemainingTiles() const {return remainingTiles;}
			
				// It is only worth dispatching this job if its tiles were not all assigned
				bool IsFullyAssigned() const {return nextTile >= tileVec.size();}
				
				//! @brief Take the Hl calculated for a certain number of tiles and 
				//  add it to Hl_total, decreasing remainingTiles by numTiles.
				//  This operation requires thread safety (handled internally)
				void Add_Hl(std::vector<real_t>& Hl, size_t const numTiles);
				
				//! @brief Wait for job completion and return final Hl
				std::vector<real_t> Get_Hl();
		};
			GCC_IGNORE_POP
		
		////////////////////////////////////////////////////////////////
		// Several variables are necessary for managing the thread pool.
		
		// Threads wait for newJob. Jobs are constructed by Hl_Job(), 
		// which notifies the threads (newJob) when the job is craeted, 
		// then notifies the manager (jobDone) when the job is complete.
		// The dtor will wait till all jobs are done before deleting the objects.
		std::condition_variable newJob, jobDone;
		
		/* Mutexes, the "talking stick" of thread communication:
		 * (dispatch_lock) is used to synchronize the dispatching of 
		 * Jobs to threads; it locks down newJobs and jobQueue.
		 * (threadSafe_lock) is used to synchronize the creation of new Jobs, 
		 * and the waiting for jobs to finish in the dtor; it locks down jobDone and activeJobs.
		 * BOTH mutexes synchronize keepAlive, the flag which keeps threads 
		 * alive and waiting. Both must be used to synchronize because
		 * keepAlive controls both new job creation (in Hl_Job()) and 
		 * dispatching of existing jobs in (Dispatch())
		*/ 
		std::mutex dispatch_lock, threadSafe_lock;
		
		// A queue of jobs, used as first-in-first out queue (FIFO); 
		// the basic FIFO is std::queue, but it lacks some useful functions.
		std::deque<std::shared_ptr<Job>> jobQueue;
		
		// To ensure that this object is thread-safe, once a job is begun
		// the object cannot be deconstructed until all active jobs are finished. 
		size_t activeJobs;
		
		// The threads are stored in the pool (a holding pen, only used during ctor and dtor)
		std::vector<std::thread> threadPool;
		
		// The thread pool is kept alive by this boolean. 
		// This is guarded by BOTH threadSafe_lock and dispatch_lock, 
		// but is only modified in the dtor.
		bool keepAlive;
		
		//! @brief Each thread requests jobs. When it has an active job, 
		// it grabs tiles till they're all gone, then adds the sum of 
		// the tiles' Hl to the job's Hl. If there are no jobs, it goes idle.
		void Hl_Thread();
		
		/*! @brief Dispatch directs the thread pool.
		 * 
		 *  It is extremely useful to encapsulate the following 
		 *  functionality into a single function: 
		 *  1. Dispatch jobs to the threads via the return value.
		 *  2. Internally prune inactive jobs from the jobQueue
		 *     (so that Hl_Job() does not have to manage this function). 
		 *  3. Blocks/wait until there are jobs (putting threads to sleep).
		 *  4. When all jobs are complete and keepAlive = false, it returns
		 *     nullptr (shared_ptr equivalent), instructing threads to exit Hl_Thread().
		*/
		std::shared_ptr<Job> Dispatch();
		
		/*! @brief Construct a new Job (e.g. the list of tiles in its tileVec),
		 *  put the Job in the jobQueue, notify the threads, and return the result.
		 * 
		 *  This function is thread-safe; it can be called by multiple threads
		 *  simultaneously without any side effects or race conditions.
		 *  All jobs requested before the dtor is called are guaranteed to finish.
		 * 
		 *  @throws throws a runtime_error if called after the dtor.
		 */
		std::vector<real_t> Hl_Job(ShapedParticleContainer const* const left_in, 
			ShapedParticleContainer const* const right_in, size_t const lMax);
	
	public:
		//! @brief Construct a thread-management object with the given pool size.
		PowerSpectrum(size_t const numThreads = maxThreads);	
		
		//! @brief Wait for all active jobs to finish, then destroy the thread pool.
		~PowerSpectrum();
	
		//! @brief Calculate the power spectrum for a set of particles
		std::vector<real_t> Hl_Obs(size_t const lMax, 
			ShapedParticleContainer const& particles);
			
		template<class particle_t>
		std::vector<real_t> Hl_Obs(size_t const lMax, 
			std::vector<particle_t> const& particles)
		{
			return Hl_Obs(lMax, ShapedParticleContainer(particles));
		}		
		
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
		
		/*! @brief Write a set of power spectra to file.
		 * 
		 *  Any Hl set which is shorter than the longest set 
		 *  will be padded with -1., a nonsense value.
		*/ 
		static void WriteToFile(std::string const& filePath, 
			std::vector<std::vector<real_t>> const& Hl_set,
			std::string const& header = "");
		
		//! @brief Write a single power spectra to file.
		static void WriteToFile(std::string const& filePath, 
			std::vector<real_t> const& Hl, 
			std::string const& header = "");
			
		
		static real_t SmearedDistance_TrackTower(real_t const r);
		
		static real_t SmearedDistance_TowerTower(real_t const r);
			
		static real_t AngularResolution(std::vector<PhatF> const& tracks,
			real_t const ff_fraction = real_t(1));
		
		static real_t AngularResolution(std::vector<PhatF> const& tracks, 
			std::vector<PhatF> const& towers, real_t const squareWidth,
			real_t const ff_fraction = real_t(1));
			
		static std::pair<std::vector<real_t>, std::vector<std::vector<real_t>>>
			AngularCorrelation(std::vector<std::vector<real_t>> const& Hl_vec, 
				size_t const zSamples = 2048);
		
		static void Write_AngularCorrelation(std::string const& filePath, 
			std::vector<std::vector<real_t>> const& Hl_vec, size_t const zSamples = 2048, 
			std::string const& header = "");
};

GCC_IGNORE_POP

#endif
