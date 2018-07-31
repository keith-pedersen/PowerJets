#include "PowerJets.hpp"
#include "PowerSpectrum.hpp"
#include "RecursiveLegendre.hpp"
#include "ShapeFunction.hpp"
#include "kdp/kdpStdVectorMath.hpp"
#include <future>

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(vec3_t const& p3):
	pHat(p3), f(p3.Mag())
{
	pHat /= f;
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(vec4_t const& p4):
	pHat(p4.p()), f(p4.x0)
{
	pHat.Normalize();
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(Jet const& jet):
	PhatF(jet.p4) {}
	
////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(real_t const px, real_t const py, real_t const pz, real_t f_in):
	pHat(px, py, pz), f(f_in)
{
	pHat.Normalize(); // Don't use f to normalize, in case particle has small mass
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(Pythia8::Particle const& particle):
	PhatF(particle.px(), particle.py(), particle.pz(), particle.e()) {}
	
////////////////////////////////////////////////////////////////////////
	
PowerSpectrum::real_t PowerSpectrum::PhatF::fInner(std::vector<PhatF> const& particles)
{
	real_t f2 = real_t(0);
	
	for(auto const& particle : particles)
		f2 += kdp::Squared(particle.f);
		
	return f2;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatFvec::PhatFvec(std::vector<PhatF> const& orig)
{
	reserve(orig.size());
	for(PhatF const& p : orig)
		this->emplace_back(p);
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::PhatFvec::reserve(size_t const reserveSize)
{
	x.reserve(reserveSize);
	y.reserve(reserveSize);
	z.reserve(reserveSize);
	
	f.reserve(reserveSize);
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::PhatFvec::clear()
{
	x.clear();
	y.clear();
	z.clear();
	
	f.clear();
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::PhatFvec::emplace_back(PhatF const& pHatF)
{
	// Assume that pHat is properly normalized
	x.emplace_back(pHatF.pHat.x1);
	y.emplace_back(pHatF.pHat.x2);
	z.emplace_back(pHatF.pHat.x3);
	
	f.emplace_back(pHatF.f);
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::PhatFvec::append(PhatFvec const& tail)
{
	x.insert(x.end(), tail.x.begin(), tail.x.end());
	y.insert(y.end(), tail.y.begin(), tail.y.end());
	z.insert(z.end(), tail.z.begin(), tail.z.end());
	
	f.insert(f.end(), tail.f.begin(), tail.f.end());
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatFvec PowerSpectrum::PhatFvec::Join
	(PhatFvec&& first, PhatFvec&& second)
{
	// Steal the first data, copy in the second (can't steal both)
	auto retVec = PhatFvec(std::move(first));
	retVec.append(second);
	return retVec;
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer::ShapedParticleContainer(std::vector<ShapedJet> const& jets):
	PhatFvec(PhatF::To_PhatF_Vec(jets))
{
	for(auto const& jet : jets)
		shapeVec.emplace_back(jet.shape.Clone());
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer::ShapedParticleContainer
	(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape):
	PhatFvec(particles)
{
	shapeVec.emplace_back(theirSharedShape.Clone());
	while(shapeVec.size() < this->size())
		shapeVec.push_back(shapeVec.back());
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer::ShapedParticleContainer
	(std::vector<vec3_t> const& particles, ShapeFunction const& theirSharedShape):
ShapedParticleContainer(PowerSpectrum::PhatF::To_PhatF_Vec(particles), theirSharedShape) {}

////////////////////////////////////////////////////////////////////////

//~ PowerSpectrum::ShapedParticleContainer::~ShapedParticleContainer()
//~ {
	//~ for(auto const shape : shapeStore)
		//~ delete shape;
//~ }

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::ShapedParticleContainer::append
	(std::vector<PhatF> const& particles, ShapeFunction const& theirSharedShape)
{
	this->PhatFvec::append(PhatFvec(particles));
	
	shapeVec.emplace_back(theirSharedShape.Clone());
	while(shapeVec.size() < this->size())
		shapeVec.push_back(shapeVec.back());
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job::Job(ShapedParticleContainer const* const left_in,
	ShapedParticleContainer const* const right_in,
	size_t const lMax_in, 
	std::vector<TileSpecs>&& tileVec_in):
tileVec(std::move(tileVec_in)), // steal the vector
nextTile(0), remainingTiles(tileVec.size()), 
left(left_in), right(right_in), lMax(lMax_in) {}

////////////////////////////////////////////////////////////////////////

bool PowerSpectrum::Job::GetTile(TileSpecs& tile)
{
	// nextTile is a std::atomic, so assigning tiles is thread-safe.
	// It would not be thread safe if we exited a condition_variable wait based upon tileIndex; 
	// however, job completion is based on tiles which are complete, not merely assigned.
	size_t const tileIndex = nextTile++;
	
	if(tileIndex < tileVec.size()) // Only assign tiles if they exists
	{
		tile = tileVec[tileIndex];
		return true; // tile was set
	}
	else return false; // tile was not set, don't use it
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job::Add_Hl(std::vector<real_t>& Hl, size_t const numTiles)
{
	// Don't waste time synchronizing if no tiles were pulled; nothing to do
	if(numTiles) 
	{
		//Synchronize Hl_total and remainingTiles (job completion)
		std::unique_lock<std::mutex> lock(jobLock);

		if(Hl_total.empty())
			// The first thread to finish simply sets it's data (std::move for quickness)
			Hl_total = std::move(Hl);
		else
			Hl_total += Hl; // Otherwise we add to the existing
			
		remainingTiles -= numTiles;
		
		// Unlock before notify to prevent hurry-up-and-wait
		lock.unlock();
	
		// If all tiles are done, notify the sub-manager (outside of lock, 
		// this could create a race to notify, but that's not a problem).
		if(remainingTiles == 0)
		{
			// It only makes sense for no active threads if the job is completed
			assert(nextTile >= tileVec.size());
			
			done.notify_one();
		}
	}
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Job::Get_Hl()
{
	//Synchronize remainingTiles (job completion)
	std::unique_lock<std::mutex> lock(jobLock);

	// Wait until there are no more threads actively working on this job.
	while(remainingTiles)
		done.wait(lock);
	
	// When the job is done, steal the vector and return
	return std::move(Hl_total);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
		
void PowerSpectrum::Hl_Thread()
{
	static constexpr size_t incrementSize = kdp::Squared(tileWidth);
	using array_t = std::array<real_t, incrementSize>;
				
	// Permanently store the arrays which are vectorized, for better optimization	
	RecursiveLegendre_Increment<array_t> Pl_computer; // Recursively calculate Pl(pHat_i * pHat_j)
	array_t fProd; // f_i * f_j (not l-dependent)
	array_t Hl_accumulate;
	std::array<real_t, tileWidth> colShapeVal;
			
	std::shared_ptr<Job> job;
		
	// Request jobs; Dispatch() will block until there are jobs, 
	// and return nullptr when it's time to die
	while((job = Dispatch()))
	{
		size_t const lMax = job->lMax; // Tell compiler that this is a constant
		bool const symmetric = (job->left == job->right); // A symmetric outer-product
		
		// Each tile adds to this, so we must zero-initialize
		std::vector<real_t> Hl_tiles(lMax, real_t(0));
		
		TileSpecs tile;
		size_t tileCount = 0; // How many tiles are analyzed inside the loop before it terminates?
		
		// Get the index of our next tile; when GetTile() returns false,
		// there are no more tiles to analyze in this job
		while(job->GetTile(tile))
		{
			++tileCount;
			
			// Development assertions
			assert(tile.row_width <= tileWidth);
			assert(tile.col_width <= tileWidth);
			if(tile.type == TileType::RIGHT) assert(not symmetric);
			if(tile.col_width < tileWidth) assert(tile.type == TileType::FINAL);
			
			// Unless it's the final tile, the row should always be full of columns.
			// For the right edge of the asymmetric product, the columns are full but the rows are not.
			// This creates a vectorization problem which we can fix by 
			// redefining which vector supplies rows and columns.
			// WARNING: We assume that this move was anticipated when the tile was defined,
			// and the row/col definitions were already swapped.
			ShapedParticleContainer const* const rows = (tile.type == TileType::RIGHT) ? job->right : job->left;
			ShapedParticleContainer const* const cols = (tile.type == TileType::RIGHT) ? job->left : job->right;
			
			// ShapeParticleContainer.shapeVec is a vector of pointers to shape functions.
			// If all the particles use the same shape function, it is the same pointer over and over.
			// Unfortunately, these shape functions are not thread-safe, 
			// so we must clone any we intend to use. 
			// We store all clones in clone_store for memory cleanup.
			// This is faster than shared_ptr; and because we only use the clones locally, 
			// it is worth the extra effort of managing the memory manually.
			std::vector<ShapeFunction*> clone_store;
			
			// But we also need to replicate left/right->shapeVec with our clones
			// (i.e. if shapeVec has the same pointer over and over, then row/colShape will as well).
			// We will do this replication in the next two scopes, 
			// which also return values that control the logic.
			std::vector<ShapeFunction*> colShape, rowShape; 
			
			// The column shape matters when there is more than one shape function in the columns;
			// otherwise, we can apply the same shape to everything in the column
			bool const colShapeMatters = [&]()
			{
				// Here we use a lambda expression so that colShapeMatters can be const-initialized
				
				// Map the original column shapes to clones
				std::map<std::shared_ptr<ShapeFunction>, ShapeFunction*> cloneMap;
				
				// First collect all unique shapes by using cloneMap as a std::set
				// (nullptr is a placeholder)
				for(size_t j = 0; j < tile.col_width; ++j)
					cloneMap[cols->shapeVec[tile.col_beg + j]] = nullptr;
				
				// Now map all unique shapes to clones, and store the clone pointers for cleanup
				for(auto shape_it = cloneMap.begin(); shape_it not_eq cloneMap.end(); ++shape_it)
					clone_store.push_back(shape_it->second = shape_it->first->Clone());
					
					
				// If there is more than one mapping in cloneMap,
				// then colShape has more than one shape, and the column shape matters
				if(cloneMap.size() > 1)
				{
					// We replicate cols->shapeVec with the clones. 
					// Use at() as sanity check, because it throws an exception if the pointer is not found
					for(size_t j = 0; j < tile.col_width; ++j) 
						colShape.push_back(cloneMap.at(cols->shapeVec[tile.col_beg + j]));
						
					return true;
				}
				else
				{
					// Otherwise there is only one shape, and it's the only one we need
					// This relies specifically on the logic inside the l-loop, 
					// and we can't do the same thing for the rows.
					colShape.push_back(cloneMap.begin()->second);
					return false;
				}
			}();
			
			// Shape "DOESN'T matter" when all the rows have the same shape, 
			// and all the columns have the same shape, so that: 
			// 	H_l = h_{l,row} * h_{l,col} * <f_row| P_l( |p_row><p_col| ) |f_col>
			// In this case, we can worry about shape after the l-loop accumulation (less FLOPS),
			// otherwise,we must address shape before we accumulate.
			bool const shapeMatters = [&]()
			{
				// See previous lambda for instructive comments; same thing here
				std::map<std::shared_ptr<ShapeFunction>, ShapeFunction*> cloneMap;
				
				for(size_t i = 0; i < tile.row_width; ++i)
					cloneMap[rows->shapeVec[tile.row_beg + i]] = nullptr;
				
				for(auto shape_it = cloneMap.begin(); shape_it not_eq cloneMap.end(); ++shape_it)
					clone_store.push_back(shape_it->second = shape_it->first->Clone());
				
				// Even if there is only one shape, we replicate the entire rows->shapeVec,
				// do to the way that we cache the row's h_l inside the l-loop
				for(size_t i = 0; i < tile.row_width; ++i)
					rowShape.push_back(cloneMap.at(rows->shapeVec[tile.row_beg + i]));
				
				// If row shape OR column shape matters, then shape matters
				return ((cloneMap.size() > 1) or colShapeMatters);
			}();
			
			// To efficiently calculating a small, symmetric outer product
			// (i.e. Hl_Jet for N=3 jets), the last tile of a symmetric outer is "ragged".
			// This means that we only calculate the lower half and diagonal.
			// Doing this creates a *small* (2%) speed hit for large outer products, 
			// but it is definitely worth it (50% faster) for very small outer products.
			// A N approaches tileWidth, ragged tiles are definitely slower, 
			// so we only use the ragged tile when less than half the tile is full.
			bool const ragged = (symmetric and (tile.type == TileType::FINAL))
				and (tile.row_width < (tileWidth / 2));
			if(ragged) assert(tile.row_width == tile.col_width); // symmetry sanity check
			
			// Edge/Final tiles do not fill the entire increment, 
			// so we restrict the inner-product inside the l-loop to only cover the filled portion
			// (with zero-filled alignment buffer).
			size_t const k_max = tileWidth * (ragged ?
				kdp::MinPartitions(kdp::GaussSum(tile.row_width), tileWidth): 
				tile.row_width);
			
			// Similarly, we only want to sum up the filled portion of Hl_accumulate.
			// However, because BinaryAccumulate must start with a power-of-2 size, 
			// we start with the smallest power-of-2 which covers the filled portion.
			size_t const sumSize = kdp::IsPowerOfTwo(k_max) ? k_max : 
				2 * kdp::LargestBit(k_max);
			assert(sumSize <= incrementSize);
			
			// Zero out the working arrays. Only necessary once per tile because 
			// each tile will fill these arrays the same way at every l in the l-loop,
			// so any necessary buffer zero will remain a zero.
			Pl_computer.z.fill(real_t(0));
			fProd.fill(real_t(0));
			Hl_accumulate.fill(real_t(0));
			// colShape is filled inside the l-loop
				
			// Fill fProd:
			// Symmetry factors accounting for un-computed tiles are applied here
			// (to double the contribution from off-diaonal tiles, simply double their f).
			// Testing reveals that three versions is actually noticeably faster,
			// which motivates the less readable code.
			
			// Only final tiles have partially full rows, so we can hard-code
			// tileWidth as the j-loop end-condition for the non-final tiles.
			if(tile.type == TileType::FINAL)
			{
				if(ragged)
				{
					size_t k = 0;
					
					for(size_t i = 0; i < tile.row_width; ++i)
					{
						for(size_t j = 0; j <= i; ++j)
						{
							// Off-diagonal elements need doubling
							real_t const symmetry = (j == i) ? real_t(1) : real_t(2);
							
							fProd[k++] = symmetry * 
								(rows->f[tile.row_beg + i] * cols->f[tile.col_beg + j]);
						}
					}
					
					// Development assertions
					assert(k == kdp::GaussSum(tile.row_width));
					assert(k < k_max);
				}
				else
				{
					for(size_t i = 0; i < tile.row_width; ++i)
						for(size_t j = 0 ; j < tile.col_width; ++j)
							fProd[i * tileWidth + j] = 
								(rows->f[tile.row_beg + i] * cols->f[tile.col_beg + j]);
				}			
			}
			else
			{
				// For symmetric outer products, it is faster to use 
				// diagonal tiles which are full, so only they do not need doubling.
				real_t const symmetry = (symmetric and (tile.type not_eq TileType::DIAGONAL)) ? 
					real_t(2) : real_t(1);
				
				for(size_t i = 0; i < tile.row_width; ++i)
					for(size_t j = 0 ; j < tileWidth; ++j) // Hard-code tileWidth for speed
						fProd[i * tileWidth + j] = symmetry * 
							(rows->f[tile.row_beg + i] * cols->f[tile.col_beg + j]);
			}
			
			// Fill pDot with the dot product of the 3-vectors:
			// Testing reveals that it is faster to do x, y, and z in separate loops
			// (since this is how vectorization works).
			// Testing also shows that 3 versions is faster, even though it's a mess.
			{
				array_t& pDot = Pl_computer.z;
				
				if(tile.type == TileType::FINAL)
				{
					if(ragged)
					{
						size_t k = 0;
						for(size_t i = 0; i < tile.row_width; ++i)
							for(size_t j = 0; j <= i; ++j)
								pDot[k++] = rows->x[tile.row_beg + i] * cols->x[tile.col_beg + j];
								
						k = 0;
						for(size_t i = 0; i < tile.row_width; ++i)
							for(size_t j = 0; j <= i; ++j)
								pDot[k++] += rows->y[tile.row_beg + i] * cols->y[tile.col_beg + j];
						
						k = 0;
						for(size_t i = 0; i < tile.row_width; ++i)
							for(size_t j = 0; j <= i; ++j)
								pDot[k++] += rows->z[tile.row_beg + i] * cols->z[tile.col_beg + j];
					}
					else
					{
						for(size_t i = 0; i < tile.row_width; ++i)
							for(size_t j = 0 ; j < tile.col_width; ++j)
								pDot[i * tileWidth + j] = 
									rows->x[tile.row_beg + i] * cols->x[tile.col_beg + j];
									
						for(size_t i = 0; i < tile.row_width; ++i)
							for(size_t j = 0 ; j < tile.col_width; ++j)
								pDot[i * tileWidth + j] += 
									rows->y[tile.row_beg + i] * cols->y[tile.col_beg + j];
									
						for(size_t i = 0; i < tile.row_width; ++i)
							for(size_t j = 0 ; j < tile.col_width; ++j)
								pDot[i * tileWidth + j] += 
									rows->z[tile.row_beg + i] * cols->z[tile.col_beg + j];
					}
				}
				else
				{
					for(size_t i = 0; i < tile.row_width; ++i)
						for(size_t j = 0 ; j < tileWidth; ++j)
							pDot[i * tileWidth + j] = 
								rows->x[tile.row_beg + i] * cols->x[tile.col_beg + j];
								
					for(size_t i = 0; i < tile.row_width; ++i)
						for(size_t j = 0 ; j < tileWidth; ++j)
							pDot[i * tileWidth + j] += 
								rows->y[tile.row_beg + i] * cols->y[tile.col_beg + j];
								
					for(size_t i = 0; i < tile.row_width; ++i)
						for(size_t j = 0 ; j < tileWidth; ++j)
							pDot[i * tileWidth + j] += 
								rows->z[tile.row_beg + i] * cols->z[tile.col_beg + j];
				}
			}
			
			// Prepare to iterate
			Pl_computer.Reset();
			
			for(size_t l = 1; l <= lMax; ++l)
			{
				// The branch is quite predictable, and saves one unused call to Pl_computer.Next()
				// The speed advantage is actually quite noticeable. 
				if(l > 1)
					Pl_computer.Next();
				assert(Pl_computer.l() == l);
				
				// Do the original calculation for Dirac delta particles
				for(size_t k = 0; k < k_max; ++k)
					Hl_accumulate[k] = Pl_computer.P_l()[k] * fProd[k];
				
				// Do we need to handle shape inside the l-loop?
				if(shapeMatters)
				{
					// If columns have different shapes, cache the value of each shape function.
					// Redundant calls to repeated shapes are still somewhat efficient,
					// as the cached value is returned (instead of recalculating).
					if(colShapeMatters)
					{
						for(size_t j = 0; j < tile.col_width; ++j)
							colShapeVal[j] = colShape[j]->hl(l);
					}
					else
						colShapeVal.fill(colShape.front()->hl(l));
					
					// The row's h_l will be cached one at a time, since rows are the outer loop
					if(ragged)
					{
						size_t k = 0;
						for(size_t i = 0; i < tile.row_width; ++i)
						{
							real_t const rowShapeVal = rowShape[i]->hl(l);
							
							for(size_t j = 0; j <= i; ++j)
								Hl_accumulate[k++] *= rowShapeVal * colShapeVal[j];
						}
					}
					else
					{
						for(size_t i = 0; i < tile.row_width; ++i)
						{
							real_t const rowShapeVal = rowShape[i]->hl(l);
							
							// Wait! If this is a final tile, it's row may not be full!
							// Doesn't matter, we are writing garbage into something we won't sum
							// (and is probably already zero, so 0 *= shit = 0). 
							// This is faster because MOST tiles are full,
							// and this solution creates less branches overall.
							for(size_t j = 0 ; j < tileWidth; ++j)
								Hl_accumulate[i * tileWidth + j] *= rowShapeVal * colShapeVal[j];
						}
					}
				}
				
				// Now that we've applied shape, we can accumulate all the terms
				real_t Hl_sum = kdp::BinaryAccumulate_Destructive(Hl_accumulate, sumSize);
				
				// If shape doesn't matter, we can handle it after we accumulate (less FLOPS)
				if(not shapeMatters)
					Hl_sum *= rowShape.front()->hl(l) * colShape.front()->hl(l);
				
				Hl_tiles[l - 1] += Hl_sum; // l = 0 is not stored
			}// end l-loop
			
			for(auto const clone : clone_store) // Delete all the shape function clones
				delete clone;
		}
			
		// No more tiles; this thread is done doing major work.
		// Since it has nothing better to do, use the thread to 
		// add to Hl_total before returning completely
		job->Add_Hl(Hl_tiles, tileCount);
	}
}

////////////////////////////////////////////////////////////////////////

std::shared_ptr<PowerSpectrum::Job> PowerSpectrum::Dispatch()
{
	// Synchronize keepAlive and jobQueue
	std::unique_lock<std::mutex> dispatch(dispatch_lock);
	
	// Some will say while(true) is a bad practice; a never-ending loop?
	// However, I feel this is the most readable way to say what this loop does.
	// In perpetuity:
	//    1. Look for jobs to dispatch, pruning those which are not dispatchable.
	//    2. If there are no jobs and we are supposed to keepAlive, sleep. 
	//    3. Only when there are no jobs and it's time to die do we kill the calling thread
	while(true)
	{
		// Each Job is assigned to every thread until all the job's tiles
		// have been assigned, at which point the Job is pruned from the queue.
		// Keep dispatching Jobs as long as they exist (even if keepAlive is false);
		// this allows all existing jobs to finish before the object deconstructs.
		while(jobQueue.size())
		{
			// Do something to the first job; either prune or dispatch
			if(jobQueue.front()->IsFullyAssigned())
				jobQueue.pop_front();
			else
				return jobQueue.front();
		}
		
		// We can now be assured that the job queue is empty, so ...
		if(keepAlive) // ... we wait for new job if instructed
			newJob.wait(dispatch); 
		else // ... or we kill the calling thread if it's time to deconstruct
			return std::shared_ptr<Job>(nullptr); // same as std::shared_ptr<Job>(), but more readable
			
		// Keeping the nullptr return in else ensures that, upon wakeup, 
		// we immediately look for new jobs, instead of checking keepAlive
	}
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PowerSpectrum(size_t const numThreads):
	activeJobs(0), keepAlive(true)
{
	threadPool.reserve(numThreads);
	
	// Populate the thread pool.
	// They will immediately find no work to do, and will await notification.
	for(size_t t = 0; t < numThreads; ++t)
		threadPool.emplace_back(&PowerSpectrum::Hl_Thread, this);
}

////////////////////////////////////////////////////////////////////////'

PowerSpectrum::~PowerSpectrum()
{
	// keepAlive is guarded by both mutexes, since it controls both Hl_Job() and Dispatch()
	std::unique_lock<std::mutex> threadSafe(threadSafe_lock);
	std::unique_lock<std::mutex> dispatch(dispatch_lock);
	
	keepAlive = false; // Stop all threads after all current jobs are complete
	dispatch.unlock(); // Unlock dispatch so threads can finish existing jobs
		
	newJob.notify_all(); // Wake up any sleeping threads
	
	while(activeJobs) // Wait for active jobs to finish
		jobDone.wait(threadSafe);
	
	// Now wait for each thread to realize (in Dispatch) that it's time to die
	for(size_t t = 0; t < threadPool.size(); ++t)
		threadPool[t].join();
}

////////////////////////////////////////////////////////////////////////'
		
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Job(
	ShapedParticleContainer const* const left, 
	ShapedParticleContainer const* const right, 
	size_t const lMax)
{
	// Synchronize the thread pool to check that keepAlive is true
	std::unique_lock<std::mutex> threadSafe(threadSafe_lock);
		
	// By using a unique_lock, threadSafe will be unlocked when this exception is thrown
	if(not keepAlive)
		throw std::runtime_error("PowerSpectrum: this object is in the process of being deconstructed!");
	
	// The thread pool is now guaranteed to persist till after the job is complete
	++activeJobs;
	threadSafe.unlock();
		
	// Return an empty vector if it's going to be empty anyway
	if((lMax == 0) or (left->size() == 0) or (right->size() == 0))
		return std::vector<real_t>();
	else
	{
		std::vector<TileSpecs> tileVec;
					
		// There are two different tiling methods, depending on symmetric or asymmetric outer products.
		// These can be combined into a single loop to make less code, 
		// but then the code becomes less readable. No time is spent here, leave it be.
		if(left == right)
		{
			for(size_t i = 0; i < left->size(); i += tileWidth)
			{
				// Detect the last row of tiles
				TileType rowType_base = ((i + tileWidth) >= left->size()) ? 
					TileType::BOTTOM : TileType::CENTRAL;
				
				// Only complete the lower-half of the outer product; doubling the result.
				for(size_t j = 0; j <= i; j += tileWidth)
				{
					size_t const row_width = (rowType_base == TileType::BOTTOM) ? 
						left->size() - i : tileWidth;
						
					TileType type = rowType_base;
					
					// Detect a diagonal tile
					if(i == j)
					{
						type = (rowType_base == TileType::BOTTOM) ? 
							TileType::FINAL : TileType::DIAGONAL;
					}
					
					// A rows are full of columns except the final tile
					size_t const col_width = (type == TileType::FINAL) ? 
						right->size() - j : tileWidth;
						
					tileVec.push_back(TileSpecs{i, row_width, j, col_width, type});
				}
			}
		}
		else
		{
			for(size_t i = 0; i < left->size(); i += tileWidth)
			{
				TileType rowType_base = ((i + tileWidth) >= left->size()) ? 
					TileType::BOTTOM : TileType::CENTRAL;
				
				for(size_t j = 0; j < right->size(); j += tileWidth)
				{
					size_t const row_width = (rowType_base == TileType::BOTTOM) ? 
						(left->size() - i) : tileWidth;
						
					TileType type = rowType_base;
					
					if((j + tileWidth) >= right->size())
					{
						type = (rowType_base == TileType::BOTTOM) ? 
							TileType::FINAL : TileType::RIGHT;
					}
					
					size_t const col_width = (type not_eq rowType_base) ? 
						(right->size() - j) : tileWidth;
						
					if(type == TileType::RIGHT)
						// Swap rows and columns so rows are full
						tileVec.push_back(TileSpecs{j, col_width, i, row_width, type});
					else
						tileVec.push_back(TileSpecs{i, row_width, j, col_width, type});
				}
			}
		}
		
		// We should have caught the three conditions that cause zero tiles
		assert(tileVec.size() > 0);
		
		std::shared_ptr<Job> job(new Job(left, right, lMax, std::move(tileVec)));
		
		// Lock dispatch to add the job to the queue
		std::unique_lock<std::mutex> dispatch(dispatch_lock);
			jobQueue.push_back(job);
		dispatch.unlock();
		
		// Wake up the threads to do their work.
		if(job->RemainingTiles() <= minTilesPerThread)
			newJob.notify_one();
		else
			newJob.notify_all();
			
		std::vector<real_t> Hl_final = job->Get_Hl();
		
		// Lock threadSafe to notify completion of this job.
		threadSafe.lock();		
			--activeJobs;
			
		/* Normally we would now unlock before notification, 
		 * to prevent a hurry-up-and-wait. But the only thing that's waiting 
		 * for jobDone is the dtor waiting for active jobs to finish.
		 * Once activeJobs == 0 and threadSafe is unlocked,
		 * if the dtor happened to spuriously wakeup before notification,
		 * it would immedietely begin killing this object.
		 * If this happens BEFORE this function notifies OR returns ... undefined behavior 
		 * (calling notify on a deconstructed CV, etc.). 
		 * This race condition is possible, so its inevitable.
		 * To prevent it, we keep threadSafe locked until the function returns.
		*/
			
		jobDone.notify_one();
		
		return Hl_final;
	}
}

////////////////////////////////////////////////////////////////////////
	
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Obs(size_t const lMax, 
	ShapedParticleContainer const& particles)
{
	return Hl_Job(&particles, &particles, lMax);
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Jet(size_t const lMax, 
	ShapedParticleContainer const& jets, std::vector<real_t> const& hl_onAxis_Filter)
{
	if(hl_onAxis_Filter.size() < lMax)
		throw std::runtime_error("PowerSpectrum::Hl_Jet: on-axis filter too short");
	
	auto Hl_vec = Hl_Job(&jets, &jets, lMax);
	 	
	for(size_t lMinus1 = 0; lMinus1 < Hl_vec.size(); ++lMinus1)
		Hl_vec[lMinus1] *= kdp::Squared(hl_onAxis_Filter[lMinus1]);
		
	return Hl_vec;
}

////////////////////////////////////////////////////////////////////////
	
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Jet(size_t const lMax, 
	std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter)
{
	return Hl_Jet(lMax, ShapedParticleContainer(jets), hl_onAxis_Filter);
}

////////////////////////////////////////////////////////////////////////
	
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Hybrid(size_t const lMax,
	std::vector<ShapedJet> const& jets_in, std::vector<real_t> const& hl_onAxis_Filter,
	ShapedParticleContainer const& particles,
	std::vector<real_t> const& Hl_Obs_in)
{
	if(hl_onAxis_Filter.size() < lMax)
		throw std::runtime_error("PowerSpectrum::Hl_Jet: on-axis filter too short");
	
	/* Because rho = 0.5(jets + particles), 
	 * H_l = 1/4 * (H_l^jets + H_l^particles + 2 * H_l^{jets, particles})
	 * This is superior than calculating a massive ShapedParticleContainer
	 * because we can reuse Hl_obs, which is likely the largest quadrant (PP below)
	 * 
	 *     J = jet, P = particle
	 * 
	 * +-------+-------------+
	 * | JJ JJ | JP JP JP JP | 
	 * | JJ JJ | JP JP JP JP | 
	 * +-------+-------------|
	 * | JP JP | PP PP PP PP | 
	 * | JP JP | PP PP PP PP | 
	 * | JP JP | PP PP PP PP | 
	 * | JP JP | PP PP PP PP | 
	 * +-------+-------------+
	 */ 
	
	std::vector<real_t> Hl;
	
	// Since this Hl is composed form 2 (or 3) separate power spectra, 
	// we can start those Hl calculations asynchronously, to better utilize the thread pool.
	std::future<std::vector<real_t>> Hl_Obs_future, Hl_jets_particles_future, Hl_jets_future;
	
	bool const recalculate_Obs = (Hl_Obs_in.size() < lMax);
	
	// Reuse Hl_Obs if possible, but if not launch the particles-particles job
	// This is probably the longest job, so we start it first to warm up the thread pool
	// (and so we construct the ShapedParticleContainer while something else is running).
	if(recalculate_Obs)
	{
		Hl_Obs_future = std::async(std::launch::async, 
		static_cast<std::vector<real_t>(PowerSpectrum::*)(size_t const,
			ShapedParticleContainer const&)>(&PowerSpectrum::Hl_Obs), this, 
			lMax, std::cref(particles));
	}
	else
		Hl.assign(Hl_Obs_in.cbegin(), Hl_Obs_in.cbegin() + lMax);	
	
	// Start the calculations involving the jets
	{
		// This is passed by REFERENCE to the Hl calculation functions, 
		// so it MUST have the lifetime of the calculations it is used for;
		// otherwise the functions will deference a deconstructed object		
		ShapedParticleContainer jets(jets_in);
		
		Hl_jets_particles_future = std::async(std::launch::async, &PowerSpectrum::Hl_Job, this,
			&jets, &particles, lMax);
			
		Hl_jets_future = std::async(std::launch::async, 
			// The compiler cannot figure out which overloaded function to call; we must help it out
			// https://stackoverflow.com/questions/27033386/stdasync-with-overloaded-functions
			static_cast<std::vector<real_t>(PowerSpectrum::*)(size_t const,
			ShapedParticleContainer const&, std::vector<real_t> const&)>(&PowerSpectrum::Hl_Jet), this,
			lMax, std::cref(jets), std::cref(hl_onAxis_Filter));
		
		// With all jobs in the queue, get the one that should come off first.
		if(recalculate_Obs)
			Hl = Hl_Obs_future.get();
			
		{
			// Assign this to a temporary spot so we can apply the detector filter
			auto Hl_jets_particles = Hl_jets_particles_future.get();
			
			// Add the doubling factor when we apply the detector filter
			for(size_t lMinus1 = 0; lMinus1 < Hl_jets_particles.size(); ++lMinus1)
				Hl_jets_particles[lMinus1] *= real_t(2) * hl_onAxis_Filter[lMinus1];
			
			Hl += Hl_jets_particles;
		}
		
		Hl += Hl_jets_future.get();
	}
			
	Hl *= real_t(0.25); // To reduce FLOPS, divide by four once
	return Hl;
}

void PowerSpectrum::WriteToFile(std::string const& filePath,
	std::vector<std::vector<real_t>> const& Hl_set, std::string const& header)
{
	std::ofstream file(filePath, std::ios::trunc);
	
	file << "#" << header << "\n";
	
	size_t const lMax = [&]()
	{
		size_t maxSize = 0;
		
		for(auto const& Hl : Hl_set)
			maxSize = std::max(maxSize, Hl.size());
			
		return maxSize;
	}();
	
	constexpr char const* l_format = "%5lu ";
	constexpr char const* Hl_format = "%.16e ";
	
	if(not file.is_open())
		throw std::ios::failure("PowerSpectrum::WriteToFile: File cannot be opened for write: " + filePath);
	{
		char buff[128];

		sprintf(buff, l_format, 0lu);
		file << buff;		
		
		for(size_t k = 0; k < Hl_set.size(); ++k)
		{
			sprintf(buff, Hl_format, 1.);
			file << buff;
		}
		
		file << "\n";
		
		for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
		{
			sprintf(buff, l_format, lMinus1 + 1);
			file << buff;		
			
			for(size_t k = 0; k < Hl_set.size(); ++k)
			{
				real_t const val = (Hl_set[k].size() > lMinus1) ? Hl_set[k][lMinus1] : -1.;
				sprintf(buff, Hl_format, val);
				file << buff;
			}
			
			file << "\n";
		}
	}
}

void PowerSpectrum::WriteToFile(std::string const& filePath,
	std::vector<real_t> const& Hl, std::string const& header)
{
	std::vector<std::vector<real_t>> const Hl_set = {Hl};
	PowerSpectrum::WriteToFile(filePath, Hl_set, header);
}

/* The next two functions use parameters determined from fits to MonteCarlo integration. 
 * See the PowerJets paper for a full explanation of these fits 
*/ 
PowerSpectrum::real_t PowerSpectrum::SmearedDistance_TrackTower(real_t const r)
{
	static constexpr real_t f0 = 0.667;
	static constexpr real_t b = 2.38;
	
	return std::pow(std::pow(f0, b) + std::pow(r, b), real_t(1)/b);	
}

PowerSpectrum::real_t PowerSpectrum::SmearedDistance_TowerTower(real_t const r)
{
	static constexpr real_t f0 = 0.641;
	static constexpr real_t b = 2.29;
	
	return std::pow(std::pow(f0, b) + std::pow(r, b), real_t(1)/b);
}

PowerSpectrum::real_t PowerSpectrum::AngularResolution(std::vector<PhatF> const& tracks, 
	real_t const ff_fraction)
{
	return AngularResolution(tracks, std::vector<PhatF>(), real_t(1), ff_fraction);
}

PowerSpectrum::real_t PowerSpectrum::AngularResolution(std::vector<PhatF> const& tracks,
	std::vector<PhatF> const& towers, real_t const squareWidth, 
	real_t const ff_fraction)
{
	if((tracks.size() + towers.size()) < 3)
	{
		if((tracks.size() + towers.size()) == 2)
			return M_PI;
		else
			return INFINITY;
	}
	else
	{	
		using angleWeight = std::pair<real_t, real_t>;
		
		std::vector<angleWeight> angleVec;		
		real_t ff = 0.;
		
		// 2 pi (1-cos(thetaR)) = (squareWidth)**2 ==>? sin(thetaR/2)**2 = (squareWidth)**2/(4*Pi)
		double twrRadius = real_t(2)*std::asin(real_t(0.5)*squareWidth/std::sqrt(M_PI));
		double twoTwrRadius = std::sqrt(real_t(2))*twrRadius; // radii add in quadrature
		
		for(size_t trk = 0; trk < tracks.size(); ++trk)
		{
			ff += kdp::Squared(tracks[trk].f);
			
			for(size_t trk_other = 0; trk_other < trk; ++trk_other)
			{
				angleVec.emplace_back(
					// No smearing between tracks; assume very well measured
					tracks[trk].pHat.InteriorAngle(tracks[trk_other].pHat),
					tracks[trk].f * tracks[trk_other].f);
			}
			
			for(size_t twr = 0; twr < towers.size(); ++twr)
			{
				angleVec.emplace_back(
					twrRadius * SmearedDistance_TrackTower(
						tracks[trk].pHat.InteriorAngle(towers[twr].pHat)/twrRadius),
					tracks[trk].f * towers[twr].f);
			}		
		}
		
		for(size_t twr = 0; twr < towers.size(); ++twr)
		{
			ff += kdp::Squared(towers[twr].f);
			
			for(size_t twr_other = 0; twr_other < twr; ++twr_other)
			{
				angleVec.emplace_back(
					twoTwrRadius * SmearedDistance_TowerTower(
						towers[twr].pHat.InteriorAngle(towers[twr_other].pHat)/twoTwrRadius),
					towers[twr].f * towers[twr_other].f);
			}
		}
		
		std::sort(angleVec.begin(), angleVec.end(), 
			[](angleWeight const& left, angleWeight const& right){return left.first < right.first;});
			
		real_t weight = real_t(0);
		real_t const weight_target = real_t(0.5) * ff * ff_fraction; // half because each distance appears twice in the expansion
		real_t geoMean = real_t(0);
		
		size_t i = 0;
		
		for(; (i < angleVec.size()) and (weight < weight_target); ++i)
		{
			geoMean += angleVec[i].second * std::log(angleVec[i].first);
			weight += angleVec[i].second;
		}
		
		assert(i < angleVec.size());	
		
		return std::exp(geoMean / weight);
	}
}

std::pair<std::vector<PowerSpectrum::real_t>, std::vector<std::vector<PowerSpectrum::real_t>>>
PowerSpectrum::AngularCorrelation(std::vector<std::vector<real_t>> const& Hl_vec, 
	size_t const zSamples)
{
	using vec_t = std::vector<real_t>;
	RecursiveLegendre_Increment<vec_t> Pl_computer;

	for(size_t i = 0; i < zSamples; ++i)
		Pl_computer.z.emplace_back(real_t(-1) + ((real_t(i) + real_t(0.5)) * real_t(2)) / real_t(zSamples));
	assert(Pl_computer.z.size() == zSamples);
	
	// Accumulate A(z) for each Hl, default emplacing l=0	
	std::vector<vec_t> A_vec(Hl_vec.size(), vec_t(zSamples, real_t(1)));
	
	size_t const lMax = [&]()
	{
		std::vector<size_t> lMax_vec;
		
		for(auto const& Hl : Hl_vec)
			lMax_vec.emplace_back(Hl.size());
			
		return *(std::max_element(lMax_vec.begin(), lMax_vec.end()));
	}();
	
	if(A_vec.size())
	{
		Pl_computer.Reset(); // l = 1
		assert(Pl_computer.l() == 1);
		
		for(size_t l = 1; l <= lMax; ++l)
		{
			real_t const twoLp1 = real_t(2*l + 1);
			
			for(size_t i = 0; i < Hl_vec.size(); ++i)
			{
				vec_t const& Hl = Hl_vec[i];
				vec_t& A = A_vec[i];
				
				real_t const C_l = twoLp1 * ((l < Hl.size()) ? Hl[l-1] : real_t(0));				
				
				for(size_t k = 0; k < zSamples; ++k)
					A[k] += C_l * Pl_computer.P_l()[k];
			}
			
			Pl_computer.Next();
		}
	}
	
	return {Pl_computer.z, A_vec};
}

void PowerSpectrum::Write_AngularCorrelation(std::string const& filePath, 
	std::vector<std::vector<real_t>> const& Hl_vec, size_t const zSamples, 
	std::string const& header)
{
	std::ofstream file(filePath, std::ios::trunc);
	char buff[1024];
	
	if(not file.is_open())
		throw std::ofstream::failure("Cannot open: <" + filePath + "> for writing");
	else
	{
		file << "# " << header << "\n";
		
		auto const series = AngularCorrelation(Hl_vec, zSamples);
		
		for(size_t i = 0; i < series.first.size(); ++i)
		{
			sprintf(buff, "%.16e", series.first[i]);
			file << buff;
			
			for(size_t k = 0; k < Hl_vec.size(); ++k)
			{
				sprintf(buff, "  %.16e", series.second[k][i]);
				file << buff;
			}
			file << "\n";
		}
	}
}

/* Previously I used two methods to launch threads in worker functions (not ctor)
	
// 1. 
std::vector<std::future<std::vector<real_t>>> threadReturn;
	
for(size_t t = 0; t < numThreads; ++t)
{
	// Note, member pointer must be &class::func not &(class::func), https://stackoverflow.com/questions/7134197/error-with-address-of-parenthesized-member-function
	// Launch with launch::async to launch thread immediately 
	threadReturn.push_back(std::async(std::launch::async, &PowerSpectrum::Hl_Thread, this));
}		

for(size_t t = 0; t < numThreads; ++t)
{
	// Get the result (get() will block until the result is ready), The return is an r-value, so we obtain it by value
	std::vector<real_t> Hl_vec_thread = threadReturn[t].get();
		
	if(t == 0) // Intialize to first increment by stealing thread's data
		Hl_vec = std::move(Hl_vec_thread);
	else
		Hl_vec += Hl_vec_thread; // The number of threads is limited, so there's no point in a binary sum.
}

// 2. 
std::thread threadStore[numThreads];

for(size_t t = 0; t < numThreads; ++t) // These threads safely add to Hl_total, a variable owned by this
	threadStore[t] = std::thread(&PowerSpectrum::DoWork_ThenAddToTotal, this);
	
for(size_t t = 0; t < numThreads; ++t)
	threadStore[t].join();
	 
*/
