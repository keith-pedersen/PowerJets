#ifndef POWER_SPECTRUM
#define POWER_SPECTRUM

#include "PowerJets.hpp"
//~ #include "SpectralPower.hpp"
#include "Pythia8/Event.h"
#include "NjetModel.hpp"
#include <atomic>

/*! @brief A class for parallel, cache-efficient calculation of power spectrum H_l
 * 
 *  Class objects manage threads during parallel execution, 
 *  and are only accessible via the class' static interface.
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
				ShapedParticleContainer() {}
			
				ShapedParticleContainer(std::vector<ShapedJet> const& jets);
				ShapedParticleContainer(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape);
				~ShapedParticleContainer();
				
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
	
		//! @brief A tile is specified by its boundaries and type
		struct TileSpecs
		{
			size_t row_beg; //! @brief The index of the first row
			size_t row_width; //! @brief The number of rows
			size_t col_beg; //! @brief The index of the first column
			size_t col_width; //! @brief The number of columns
			TileType type;
		};
		
		// By default, the "left" supplies the rows and the "right" the columns.
		// However, this is reverse for RIGHT tiles, so that only the FINAL 
		// tiles have half-full rows.
		ShapedParticleContainer const* left;
		ShapedParticleContainer const* right;
		
		// A list of tiles which need calculation
		std::vector<TileSpecs> tileVec;
		// A thread safe iterator used to assign the next tile to each thread
		std::atomic<size_t> nextTile;
		
		std::vector<real_t> Hl_total; // The total Hl
		std::mutex returnLock; // The lock to modify Hl_total		
		
		size_t const lMax; //! @brief Calculate H_l form l = 1 -- lMax
		
		//! @brief Keeps grabbing tiles till they're all gone, 
		// returning the sum of their H_l contributions.
		std::vector<real_t> Hl_Thread();
		
		//! @brief Call Hl_Thread, then add the result to Hl_total
		//  (since that thread now has nothing better to do).
		void DoWork_ThenAddToTotal();		
		
		//! @brief Given a constructed object, launch the threads and combine their results
		std::vector<real_t> Launch();
		
		//! @brief Construct a thread management object that will calculate H_l when launched
		PowerSpectrum(ShapedParticleContainer const* const left_in, 
			ShapedParticleContainer const* const right_in, size_t lMax_in);
	
	public:		
		//! @brief Calculate the power spectrum for a set of particles
		static std::vector<real_t> Hl_Obs(size_t const lMax, 
			ShapedParticleContainer const& particles);
			
		~PowerSpectrum() {}
		
		/*! @brief Calculate the power spectrum for a set of jets
		 * 
		 *  hl_onAxis_Filter applies the filter of the detector elements.
		*/ 
		static std::vector<real_t> Hl_Jet(size_t const lMax, 
			ShapedParticleContainer const& jets, std::vector<real_t> const& hl_onAxis_Filter);
			
		//! @brief Calculate the power spectrum for a set of jets
		static std::vector<real_t> Hl_Jet(size_t const lMax, 
			std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter);
		
		//! @brief Calculate the hybrid power spectrum for rho = 0.5*(jets + particles)
		static std::vector<real_t> Hl_Hybrid(size_t const lMax,
			std::vector<ShapedJet> const& jets_in, std::vector<real_t> const& hl_onAxis_Filter,
			ShapedParticleContainer const& particles,
			std::vector<real_t> const& Hl_Obs_in = std::vector<real_t>());
};

#endif
