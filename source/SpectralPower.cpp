#include "SpectralPower.hpp"
#include "SelfOuterU.hpp"
#include "helperTools.hpp"
#include <array>
#include <future>

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

std::vector<SpectralPower::PhatF> SpectralPower::PhatF::PythiaToPhatF
	(std::vector<Pythia8::Particle> const& original)
{
	double totalE = 0.;
	
	std::vector<PhatF> newVec;
	
	for(Pythia8::Particle const& particle : original)
	{
		newVec.emplace_back(particle);
		totalE += particle.e();
	}
		
	for(auto& pHatF : newVec)
		pHatF.f /= totalE;
		
	return newVec;
}

////////////////////////////////////////////////////////////////////////

SpectralPower::PhatFvec::PhatFvec(std::vector<PhatF> const& orig,
	bool const normalize)
{
	reserve(orig.size());
	for(PhatF const& p : orig)
	{
		this->emplace_back(p.pHat, p.f, normalize);
	}
}

////////////////////////////////////////////////////////////////////////

void SpectralPower::PhatFvec::reserve(size_t const reserveSize)
{
	x.reserve(reserveSize);
	y.reserve(reserveSize);
	z.reserve(reserveSize);
	
	f.reserve(reserveSize);
}

////////////////////////////////////////////////////////////////////////

void SpectralPower::PhatFvec::clear()
{
	x.clear();
	y.clear();
	z.clear();
	
	f.clear();
}

////////////////////////////////////////////////////////////////////////

void SpectralPower::PhatFvec::emplace_back
	(vec3_t const& pHat, real_t const f_in, bool const normalize)
{
	// Not the most efficient way to normalize, but not the killer operation
	real_t const norm = (normalize ? pHat.Mag() : 1.);
	x.emplace_back(pHat.x1/norm);
	y.emplace_back(pHat.x2/norm);
	z.emplace_back(pHat.x3/norm);
	
	f.emplace_back(f_in);
}

////////////////////////////////////////////////////////////////////////

SpectralPower::PhatFvec SpectralPower::PhatFvec::Join
	(PhatFvec&& first, PhatFvec&& second)
{
	// Steal the data in first, using the move ctor
	PhatFvec joined(std::move(first));
	
	// Append second
	joined.x.insert(joined.x.end(), second.x.begin(), second.x.end());
	joined.y.insert(joined.y.end(), second.y.begin(), second.y.end());
	joined.z.insert(joined.z.end(), second.z.begin(), second.z.end());
	
	joined.f.insert(joined.f.end(), second.f.begin(), second.f.end());
	
	return joined;
}

////////////////////////////////////////////////////////////////////////

//~ void SpectralPower::PhatFvec::OuterProduct
	//~ (std::vector<real_t>& pDot, std::vector<real_t>& fProd) const
//~ {
	//~ // The weight of each dot product is controlled by f[i]*f[j]
	//~ TiledSymmetricOuterU<real_t, Equals, tileWidth>(f, fProd, 2.);
	//~ // Off diagonal terms are only calculated once, so double ^ their contribution
	
	//~ // We do the outer product of a dot product
	//~ // So calulate x products, then add y and z products
	//~ // No offDiagonal factor, because f controls the weight
	//~ TiledSymmetricOuterU<real_t, Equals, tileWidth>(x, pDot, 1.);
	//~ TiledSymmetricOuterU<real_t, PlusEquals, tileWidth>(y, pDot, 1.);
	//~ TiledSymmetricOuterU<real_t, PlusEquals, tileWidth>(z, pDot, 1.);
	
	//~ assert(fProd.size() == pDot.size());
//~ }

////////////////////////////////////////////////////////////////////////

size_t SpectralPower::Outer_Increment::Setup(PhatFvec const& source)
{
	assert(source.x.size() == source.f.size());
	
	// Safely lock the syncLock until we return (in case an exception is thrown)
	std::unique_lock<std::mutex> safeLock(syncLock); 
	
	// All of these should return the same size, but we only need one return
	xOuter.Setup(source.x, 1.);
	yOuter.Setup(source.y, 1.);
	zOuter.Setup(source.z, 1.);
	
	return fOuter.Setup(source.f, 2.);
}

////////////////////////////////////////////////////////////////////////

size_t SpectralPower::Outer_Increment::Next(std::array<real_t, incrementSize>& pDot_local, 
	std::array<real_t, incrementSize>& fProd_local)
{
	// Safely lock the syncLock until we return (in case an exception is thrown)
	std::unique_lock<std::mutex> safeLock(syncLock); 
		
	// All of these should return the same size, but we only need one return
	xOuter.Next(pDot_local);
	yOuter.Next(pDot_local);
	zOuter.Next(pDot_local);
	
	size_t const lengthSet = fOuter.Next(fProd_local);
	assert(lengthSet <= Outer_Increment::incrementSize);
	
	return lengthSet;
}

////////////////////////////////////////////////////////////////////////

SpectralPower::Settings const& SpectralPower::UpdateSettings(QSettings const& parsedSettings)
{
	settings = Settings(parsedSettings);
	return settings;
}

////////////////////////////////////////////////////////////////////////

std::vector<SpectralPower::real_t> SpectralPower::operator()
	(PhatFvec const& particles, size_t const lMax, 
	size_t const numThreads_requested)
{
	settings.lMax = lMax;
	
	size_t const numIncrements = outer.Setup(particles);
	
	// If numThreads_requested is supplied, use that, otherwise use stored setting
	size_t const maxThreads = 
		((numThreads_requested == 0) ? settings.maxThreads : numThreads_requested);
	
	// Use no more threads than the number required to ensure that
	// each thread will have at least settings.minIncrements.
	// Note that we don't use MinPartitions(numIncrements, minIncrements), 
	// because we'd rather have extra increments than extra threads.
	size_t const numThreads = std::min(maxThreads, numIncrements/settings.minIncrements);
	
	std::vector<real_t> H_l_vec; // We return a vector of H_l. 
	
	if(numThreads <= 1) // if numThreads == 0 (numIncrements  < minIncrements), use 1 thread.
		H_l_vec = H_l_threadedIncrement();
	else
	{
		// We add H_l_vec from multiple threads, so initialize to zero
		// H_l_vec[0] => l=1, so size is lMax.
		H_l_vec.assign(settings.lMax, 0.);
		
		// We need to use std::future for each thread's return value
		std::vector<std::future<std::vector<real_t>>> threadReturn;
		
		// Create/launch all threads and bind their return value
		for(size_t i = 0; i < numThreads; ++i)
		{
			// Note, member pointer must be &class::func not &(class::func)
			// https://stackoverflow.com/questions/7134197/error-with-address-of-parenthesized-member-function
			threadReturn.push_back(
				std::async(std::launch::async, &SpectralPower::H_l_threadedIncrement, this));
		}
	
		for(auto& ret : threadReturn)
		{
			// Get the result (get() will block until the result is ready)
			std::vector<real_t> const& H_l_vec_thread = ret.get();
			
			// Go through each H_l returned and add it to the running sum.
			// The number of threads is limited, so there's no point in a binary sum.
			for(size_t lMinus1 = 0; lMinus1 < settings.lMax; ++lMinus1)
				H_l_vec[lMinus1] += H_l_vec_thread[lMinus1];
		}
	}
		
	if(settings.lFactor) // Add (2l + 1) prefix
	{
		// index 0 contains l=1, so the prefix is 2*(lMinus1+1)+1 = 2*lMinus1 + 3
		for(size_t lMinus1 = 0; lMinus1 < settings.lMax; ++lMinus1)
			H_l_vec[lMinus1] *= real_t(2*lMinus1 + 3);
	}
	
	if(settings.nFactor) // Divide by <f|f>, to normalize for multiplicity
	{
		real_t f2sum;
		
		// Calculate f2sum using a binary reduction. This limits floating-point error
		// with the same algorithm that H_l_vec used
		{
			std::vector<real_t> fCopy = particles.f;
			for(real_t& f : fCopy)
				f = f*f;
				
			f2sum = BinaryAccumulate_Destructive(fCopy);
		}
		
		for(real_t& H_l : H_l_vec)
			H_l /= f2sum;
	}
	
	return H_l_vec;
}

////////////////////////////////////////////////////////////////////////

std::vector<typename SpectralPower::real_t> SpectralPower::H_l_threadedIncrement()
{
	// Worker arrays for the recursive P_l loop. std::array improves the 
	// compilers ability to auto-vectorize (fixed width, good alignment)
	using incrementArray_t = std::array<real_t, Outer_Increment::incrementSize>;
	
	// Allocate a vector of H_l to return, which we constantly add to (so initialize to zero).
	// H_l[0] actually corresponds to l == 1 (because H_0 = 1 always), so allocate lMax.
	std::vector<real_t> H_l_vec(settings.lMax, 0.);
	
	if(H_l_vec.size())
	{
		// As we process the P_l recursion in the l-loop,
		// we need to swap Pl_this -> Pl_last, Pl_next -> Pl_this.
		// But std::array::swap is VERY slow (versus std::vector), 
		// b/c its an element-wise swap (versus swapping the std::vector.data() pointer). 
		// The solution: create 3 x Pl worker arrays which we alias with pointers, 
		// then swap the pointers (essentially mimic std::vector swap,
		// just using a fixed width array(for better compiler optimization, 
		// since we know exactly how long each increment will be).
		incrementArray_t
			pDot_local, fProd_local, // increments of pDot and fProd, filled by outer
			Pl_A, Pl_B, Pl_C, // shere Pl_last, Pl_this and Pl_next will be stored
			H_l_accumulate; // the elements (P_l * fProd)_k, which we sum
		
		// Named aliases for the three P_l arrays.
		// Use __restrict to tell the compiler that they'll never collide (for optimization).
		// (__restrict works for GCC, Clang, VCC and Intel, while __restric__ is just GCC and Clang).
		incrementArray_t* __restrict Pl_last = &Pl_A;
		incrementArray_t* __restrict Pl_this = &Pl_B;
		incrementArray_t* __restrict Pl_next = &Pl_C;
		
		// IMPORTANT: before we begin, we MUST initialize Pl_last,
		// even though every increment uses Pl_last_coeff = 0. in the first iteration.
		// This is because 0.*nan -> nan. so if Pl_last is not initialized, 
		// a few elements can often randomly equal nan (which uses a 
		// special exponent) and this nan will contaminate the final H_l.
		// After the first increment, every Pl array will only contain numbers,
		// so we don't have to worry about nan again.
		Pl_last->fill(0.);
		
		size_t lengthSet; // The length of pDot and fProd set by outer.Next(pDot, fProd)
		
		// outer.Next() fills increments of pDot and fProd. 
		// The length of pDot and fProd which are set is returned by Next().
		// The last increment is not full, and is padded with zeroes by outer.Next(), 
		// which ensures that the last elements of S_l_accumulate are also zero.
		// Keep going until the lengthSet == 0 (using bool(0) = false, bool(x>0) == true).
		while(bool(lengthSet = outer.Next(pDot_local, fProd_local)))
		{
			assert(lengthSet <= Outer_Increment::incrementSize);
			
			// l corresponds to Pl_this, with each iteration of the loop setting Pl_next.
			// The "coeff" is the numerical pre-factor in the P_l recursion (e.g. (2l + 1))
			// We start the loop with l = 0. 
			real_t Pl_next_coeff = 1.; // (l + 1)
			real_t Pl_this_coeff = 1.; // (2l + 1)
			real_t Pl_last_coeff = 0.; // (-l)
			
			Pl_this->fill(1.); // P_0(x) = 1
			// No need to init Pl_last, provided it's not nan
			
			//~ size_t const loopStop = 
				//~ (lengthSet == Outer_Increment::incrementSize) ? Outer_Increment::incrementSize : 
				//~ LargestBit(lengthSet) * 2; // Last increment, calculate till power-of-2 alignment 
			
			//~ assert(loopStop <= Outer_Increment::incrementSize);
			//~ assert(loopStop >= lengthSet);
			//~ assert(IsPowerOfTwo(loopStop));
				
			for(real_t& H_l : H_l_vec) // Recursively set H_l
			{
				// NOTE: gcc won't auto-vectorize if the iterator has a 
				// smaller data type than the data type of the loop bound, 
				// because it can't ensure that the loop will terminate 
				// (from iterator overflow). Must use size_t.
				for(size_t i = 0; i < Outer_Increment::incrementSize; ++i)
				{
					// To vectorize this loop, we can't try to manually manage registers 
					// (i.e. by storing each Pl_next_i to a real_t const)
					(*Pl_next)[i] = 
						(Pl_this_coeff * pDot_local[i] * (*Pl_this)[i] 
						+ Pl_last_coeff * (*Pl_last)[i])/Pl_next_coeff;
					
					H_l_accumulate[i] = (*Pl_next)[i] * fProd_local[i];
				}
							
				H_l += BinaryAccumulate_Destructive(H_l_accumulate);//, loopStop);
				
				Pl_next_coeff += 1.; // (l + 1)				
				Pl_this_coeff += 2.; // (2l + 1)								
				Pl_last_coeff -= 1.; // (-l)
				
				{
					incrementArray_t* const Pl_newNext = Pl_last;
					Pl_last = Pl_this; // Pl_this -> Pl_last
					Pl_this = Pl_next; // Pl_next -> Pl_this
					Pl_next = Pl_newNext; // Pl_last -> Pl_next
				}
			}// Done l-loop
		}// Done increment loop
	}// Done with outer
	
	return H_l_vec;
}
