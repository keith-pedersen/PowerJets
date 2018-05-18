#include "PowerJets.hpp"
#include "PowerSpectrum.hpp"
#include "RecursiveLegendre.hpp"
#include "ShapeFunction.hpp"
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
		
void PowerSpectrum::Hl_Thread()
{
	static constexpr size_t incrementSize = kdp::Squared(tileWidth);
	using array_t = std::array<real_t, incrementSize>;
				
	RecursiveLegendre_Increment<array_t> Pl_computer; // Calculate Pl recursively
	array_t fProd;
	array_t Hl_accumulate;
	std::array<real_t, tileWidth> colShapeVal;
	std::vector<real_t> Hl_job;
	
	std::unique_lock<std::mutex> lock(syncLock); // Acquire a lock to synchronize communication
	lock.unlock(); // Immediately unlock, not needed yet
	
	while(keepAlive) // Keep this thread alive as long as instructed
	{
		// Each tile adds to this, so we must zero-initialize
		Hl_job.assign(lMax_internal, real_t(0));
		bool const symmetric = (left == right); // All tiles share same symmetry
					
		size_t tileIndex;
		
		// Get the index of our next tile
		while((tileIndex = nextTile++) < tileVec.size())
		{
			auto const& tile = tileVec[tileIndex];
			
			// Development assertions
			assert(tile.row_width <= tileWidth);
			assert(tile.col_width <= tileWidth);
			if(tile.type == TileType::RIGHT) assert(not symmetric);
			if(tile.col_width < tileWidth) assert(tile.type == TileType::FINAL);
			
			// Unless it's the final tile, the row should always be full of columns.
			// For the right edge of the asymmetric product, the columns are full but the rows are not.
			// We can fix this problem by redefining which vector supplies rows and columns.
			// WARNING: We assume that this move was anticipated by the PowerSpectrum constructor, 
			// and it already swapped row/col definitions in our TileSpecs		
			ShapedParticleContainer const* const rows = (tile.type == TileType::RIGHT) ? right : left;
			ShapedParticleContainer const* const cols = (tile.type == TileType::RIGHT) ? left : right;
			
			// ShapeParticleContainer.shapeVec is a vector of pointers to shape functions.
			// If all the particles use the same shape function, it is the same pointer over and over.
			// Unfortunately, these shape functions are not thread safe, 
			// so we must clone any we intend to use. 
			// We store all clones in clone_store for memory cleanup.
			// This is faster than shared_ptr, and because we only use the clones locally, 
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
			
			for(size_t l = 1; l <= lMax_internal; ++l)
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
				
				Hl_job[l - 1] += Hl_sum; // l = 0 is not stored
			}// end l-loop
			
			for(auto const clone : clone_store) // Delete all the shape function clones
				delete clone;
		}
		// No more tiles; this thread is done doing major work.
		// Since it has nothing better to do, use the thread to 
      // accumulate to Hl_total before returning completely
		
		// BEGIN SYNCHRONIZATION
	
		// Lock access to Hl_total. A std::atomic won't work because we
		// need to check status, then perform a conditional action.
		lock.lock();
		
		if(Hl_total.empty())
			// The first thread to finish simply sets it's data (std::move for quickness)
			Hl_total = std::move(Hl_job);
		else
			Hl_total += Hl_job; // Otherwise we add to the existing
		
		++idle; // All information has been transmitted, this thread is no longer active	
		
		// Notify the manger of job completion; but since we own the syncLock, 
		// the manager can't wake up until this thread begins its wait
		jobDone.notify_one(); 
		
		// We now begin waiting for the next job (releasing syncLock)
		// The while loop catches spurious wakeups (which are real, and were detected via assertions),
		// a possibility warned about at http://www.cplusplus.com/reference/condition_variable/condition_variable/wait/.
		// A second "predicate" argument to wait() will not work, 
		// because it requires a callable object or function with no arguments, 
		// whereas we need to access private information.
		// However, the predicate is functionally equivalent to this loop.
		do 
			newJob.wait(lock);
		while(not ValidWakeup());
		
		// Once awoken, this thread is longer idle
		--idle;
		
		// END SYNCHRONIZATION (must unlock to prevent deadlock)
		lock.unlock();
	}
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PowerSpectrum(size_t const numThreads):
	left(nullptr), right(nullptr), 
	lMax_internal(0), nextTile(0), 
	idle(0), 
	keepAlive(true)
{
	// Lock down the threads 
	std::unique_lock<std::mutex> lock(syncLock);
	threadPool.reserve(numThreads);
	
	// Populate the thread pool.
	// They will immediately find no work to do, and will await notification.
	for(size_t t = 0; t < numThreads; ++t)
		threadPool.emplace_back(&PowerSpectrum::Hl_Thread, this);
		
	// Wait until all threads are idle and ready for work before returning
	do
		jobDone.wait(lock);
	while(idle < threadPool.size());
}

////////////////////////////////////////////////////////////////////////'

PowerSpectrum::~PowerSpectrum()
{
	std::unique_lock<std::mutex> safety(threadSafe); // Probably unnecessary
	assert(idle == threadPool.size());
	
	// Tell the threads it's time to stop
	keepAlive = false;
	newJob.notify_all();
	
	// Gracefully decommission the threads
	for(size_t t = 0; t < threadPool.size(); ++t)
		threadPool[t].join();
}

////////////////////////////////////////////////////////////////////////'
		
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl(
	ShapedParticleContainer const* const left_in, 
	ShapedParticleContainer const* const right_in, 
	size_t const lMax)
{
	// Prevent other threads from talking to the threads until this thread returns
	std::unique_lock<std::mutex> safety(threadSafe);
	
	// Prepare to synchronize threads
	std::unique_lock<std::mutex> lock(syncLock);
	
	// All threads should be idle because the last job was properly finished. 
	// This assertion caught spurious wakeups
	assert(idle == threadPool.size());
	
	// Reset the outer product
	left = left_in;
	right = right_in;
	
	lMax_internal = lMax;
	
	// Reset the bookkeeping
	tileVec.clear();
	nextTile = 0;
	Hl_total.clear();
	
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
	
	// The tiles are specified; Wake up the threads to do their work.
	if(tileVec.size() <= minTilesPerThread)
		newJob.notify_one();
	else
		newJob.notify_all();
	
	// Wait until there are no more active threads
	do
		jobDone.wait(lock);
	while(idle < threadPool.size());
	
	return Hl_total;
}

////////////////////////////////////////////////////////////////////////
	
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Obs(size_t const lMax, 
	ShapedParticleContainer const& particles)
{
	return Hl(&particles, &particles, lMax);
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Jet(size_t const lMax, 
	ShapedParticleContainer const& jets, std::vector<real_t> const& hl_onAxis_Filter)
{
	if(hl_onAxis_Filter.size() < lMax)
		throw std::runtime_error("PowerSpectrum::Hl_Jet: on-axis filter too short");
	
	auto Hl_vec = Hl(&jets, &jets, lMax);
	 	
	for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
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
	
	std::vector<real_t> Hl_vec;
	
	// Reuse Hl_Obs if possible, but if not recalculate it
	if(Hl_Obs_in.size() < lMax)
		Hl_vec = Hl_Obs(lMax, particles);
	else
		Hl_vec.assign(Hl_Obs_in.cbegin(), Hl_Obs_in.cbegin() + lMax);
	
	{
		ShapedParticleContainer jets(jets_in);
		
		{
			auto Hl_jets_particles = Hl(&jets, &particles, lMax);
		
			// Add the doubling factor to this filter step
			for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
				Hl_jets_particles[lMinus1] *= real_t(2) * hl_onAxis_Filter[lMinus1];
				
			Hl_vec += Hl_jets_particles;
		}
		
		Hl_vec += Hl_Jet(lMax, jets, hl_onAxis_Filter);
	}
		
	Hl_vec *= real_t(0.25);		
	return Hl_vec;
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
