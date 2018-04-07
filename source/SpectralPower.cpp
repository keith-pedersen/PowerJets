#include "SpectralPower.hpp"
#include "SelfOuterU.hpp"
#include "NjetModel.hpp"
#include "kdp/kdpTools.hpp"
#include "RecursiveLegendre.hpp"
#include <array>
#include <future>

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

std::vector<SpectralPower::PhatF> SpectralPower::PhatF::To_PhatF_Vec
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

std::vector<SpectralPower::PhatF> SpectralPower::PhatF::To_PhatF_Vec
	(std::vector<vec3_t> const& originalVec)
{
	std::vector<PhatF> convertedVec;
	
	real_t totalE = real_t(0);
	for(auto const& original : originalVec)
	{
		real_t const mag = original.Mag();
		totalE += mag;
		convertedVec.emplace_back(original, mag, true);
	}
	
	for(auto& converted : convertedVec)
		converted.f /= totalE;
		
	return convertedVec;
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
		// We need to use std::future for each thread's return value
		std::vector<std::future<std::vector<real_t>>> threadReturn;
		
		// Create/launch all threads and bind their return value
		for(size_t t = 0; t < numThreads; ++t)
		{
			// Note, member pointer must be &class::func not &(class::func)
			// https://stackoverflow.com/questions/7134197/error-with-address-of-parenthesized-member-function
			threadReturn.push_back(
				std::async(std::launch::async, &SpectralPower::H_l_threadedIncrement, this));
		}
	
		for(size_t t = 0; t < numThreads; ++t)
		{
			// Get the result (get() will block until the result is ready)
			// The return is an r-value, so we obtain it by value
			std::vector<real_t> H_l_vec_thread = threadReturn[t].get();
			
			if(t == 0) // Intialize to first increment by stealing thread's data
				H_l_vec = std::move(H_l_vec_thread);
			else
			{
				// Go through each H_l returned and add it to the running sum.
				// The number of threads is limited, so there's no point in a binary sum.
				for(size_t lMinus1 = 0; lMinus1 < settings.lMax; ++lMinus1)
					H_l_vec[lMinus1] += H_l_vec_thread[lMinus1];
			}
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
				
			f2sum = kdp::BinaryAccumulate_Destructive(fCopy);
		}
		
		for(real_t& H_l : H_l_vec)
			H_l /= f2sum;
	}
	
	return H_l_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::operator()
	(std::vector<vec3_t> const& particles, size_t const lMax, 
	size_t const numThreads_requested)
{
	//~ PhatFvec converted;
	
	//~ std::vector<real_t> magVec;
	//~ magVec.reserve(particles.size());
	
	//~ real_t totalE = real_t(0);
	//~ for(auto const& particle : particles)
	//~ {
		//~ real_t const mag = particle.Mag();
		//~ magVec.push_back(mag);
		//~ totalE += mag;
	//~ }
	
	//~ for(size_t i = 0; i < particles.size(); ++i)
		//~ converted.emplace_back(particles[i] / magVec[i], magVec[i] / totalE, false);
	
	return (*this)(PhatF::To_PhatF_Vec(particles), lMax, numThreads_requested);
}

void SpectralPower::Write_Hl_toFile(std::string const& filePath,
	std::vector<vec3_t> const& particles, 
	size_t const lMax, size_t const numThreads_requested)
{
	std::ofstream file(filePath, std::ios::trunc);
	
	auto constexpr formatString = "%4lu %.16e\n";
	
	if(not file.is_open())
		throw std::ios::failure("File cannot be opened for write: " + filePath);
	{
		auto const H = (*this)(particles, lMax, numThreads_requested);
		
		char buff[1024];
		
		sprintf(buff, formatString, 0lu, 1.);
		file << buff;
		
		for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
		{
			sprintf(buff, formatString, lMinus1 + 1, H[lMinus1]);
			file << buff;
		}
	}	
}

////////////////////////////////////////////////////////////////////////

std::vector<typename SpectralPower::real_t> SpectralPower::H_l_threadedIncrement()
{
	// Allocate a vector of H_l to return, which we constantly add to (so initialize to zero).
	// H_l[0] actually corresponds to l == 1 (because H_0 = 1 always), so allocate lMax.
	std::vector<real_t> H_l_vec(settings.lMax, 0.);
	
	if(H_l_vec.size())
	{
		RecursiveLegendre<real_t, Outer_Increment::incrementSize> Pl_computer;
		
		using incrementArray_t = decltype(Pl_computer)::incrementArray_t;	
		
		incrementArray_t 
			fProd_increment, // increments of fProd, filled by outer
			H_l_accumulate; // the elements (P_l * fProd)_k, which we sum
		
		size_t lengthSet; // The length of pDot and fProd set by outer.Next(pDot, fProd)
		
		// outer.Next() fills increments of pDot and fProd.
		// We fill pDot directly into Pl_computer.
		// WARNING must call Pl_computer.Reset() to reset l = 0. 
		// The length of pDot and fProd which are set is returned by Next().
		// The last increment is not full, and is padded with zeroes by outer.Next(), 
		// which ensures that the last elements of H_l_accumulate are also zero.
		// Keep going until the lengthSet == 0 (using bool(0) = false, bool(x>0) == true).
		while(bool(lengthSet = outer.Next(Pl_computer.z, fProd_increment)))
		{
			Pl_computer.Reset(); 
				
			for(real_t& H_l : H_l_vec) // Recursively set H_l
			{
				auto const P_l_increment = Pl_computer.Next(); // Get the next P_l
				
				// Do a dot product with fProd
				for(size_t i = 0; i < Outer_Increment::incrementSize; ++i)
					H_l_accumulate[i] = P_l_increment[i] * fProd_increment[i];
							
				H_l += kdp::BinaryAccumulate_Destructive(H_l_accumulate);
			}// Done l-loop
		}// Done increment loop
	}// Done with outer
	
	return H_l_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Power_Extensive(size_t const lMax,
	std::vector<PhatF> const& tracks, std::vector<real_t> const& h_OnAxis_tracks, 
	std::vector<PhatF> const& towers, std::vector<real_t> const& h_OnAxis_towers)
{
	auto H_l_vec = Power_Extensive_SelfTerm(lMax, tracks, h_OnAxis_tracks);
	H_l_vec += Power_Extensive_SelfTerm(lMax, towers, h_OnAxis_towers);
	H_l_vec += Power_Extensive_SubTerm(lMax, tracks, h_OnAxis_tracks, towers, h_OnAxis_towers)*real_t(2);
	
	return H_l_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Power_Extensive_SelfTerm(size_t const lMax,
	std::vector<PhatF> const& particles, std::vector<real_t> const& h_OnAxis)
{
	// Redundant caclculation of self terms, but good enough for now.
	return Power_Extensive_SubTerm(lMax, particles, h_OnAxis, particles, h_OnAxis);
}

std::vector<SpectralPower::real_t> SpectralPower::Power_Extensive_SubTerm(size_t const lMax,
	std::vector<PhatF> const& left, std::vector<real_t> const& h_OnAxis_left, 
	std::vector<PhatF> const& right, std::vector<real_t> const& h_OnAxis_right)
{
	// Allocate a vector of H_l to return, which we constantly add to (so initialize to zero).
	// H_l[0] actually corresponds to l == 1 (because H_0 = 1 always), so allocate lMax.
	std::vector<real_t> H_l_vec(lMax, 0);
	size_t constexpr incrementSize = 64;
	
	if(H_l_vec.size())
	{
		assert((h_OnAxis_left.size() - 1) >= lMax);
		assert((h_OnAxis_right.size() - 1) >= lMax);
		
		auto const lFactor = [&](){
			// Assume h_OnAxis starts with l=0, which we don't want
			assert(h_OnAxis_left.front() == real_t(1));
			assert(h_OnAxis_right.front() == real_t(1));
			
			std::vector<real_t> hMul(h_OnAxis_left.begin() + 1, h_OnAxis_left.begin() + lMax + 1);
			for(size_t l = 0; l < hMul.size(); ++l)
				hMul[l] *= h_OnAxis_right[l + 1];
				
			return hMul;}();
		assert(lFactor.size() == lMax);
	
		RecursiveLegendre<real_t, incrementSize> Pl_computer;
		
		using incrementArray_t = decltype(Pl_computer)::incrementArray_t;	
		
		incrementArray_t 
			fProd_increment, // increments of fProd, filled by outer
			H_l_accumulate; // the elements (P_l * fProd)_k, which we sum
		
		size_t k = 0;
		
		for(size_t i = 0; i < left.size(); ++i)
		{
			for(size_t j = 0; j < right.size(); ++j)
			{
				fProd_increment[k] = left[i].f * right[j].f;
				Pl_computer.z[k] = left[i].pHat.Dot(right[j].pHat);
				++k;		
				
				// Fill the final increment with zeros
				if((i == (left.size() - 1)) and (j == (right.size() - 1)))
				{
					while(k < incrementSize)
					{
						fProd_increment[k] = real_t(0);
						Pl_computer.z[k] = real_t(0);
						++k;
					}
				}
				
				if(k == incrementSize)
				{
					Pl_computer.Reset();
					assert(Pl_computer.l() == 0);
				
					for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
					{
						auto const& P_l_increment = Pl_computer.Next(); // Get the next P_l
						//~ assert((lMinus1 + 1) == Pl_computer.l());
						
						// Do a dot product with fProd
						for(size_t m = 0; m <incrementSize; ++m)
							H_l_accumulate[m] = P_l_increment[m] * fProd_increment[m];
									
						H_l_vec[lMinus1] += lFactor[lMinus1] * 
							kdp::BinaryAccumulate_Destructive(H_l_accumulate);
					}// Done l-loop
					
					k = 0; // Start the next increment
				}
			}
		}
	}
	
	return H_l_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Power_Jets(size_t const lMax,
	std::vector<ShapedJet> const& jets, std::vector<real_t> const& detectorFilter)
{
	std::vector<real_t> H_l_vec(lMax, 0);
	size_t constexpr incrementSize = 64;
	
	RecursiveLegendre<real_t, incrementSize> Pl_computer;
	using incrementArray_t = decltype(Pl_computer)::incrementArray_t;	
	
	std::vector<std::vector<real_t>> h_OnAxis;
	Pl_computer.z.fill(real_t(0));
		
	{
		size_t k = 0;
		
		for(size_t i = 0; i < jets.size(); ++i)
		{
			vec3_t pHat_i = vec3_t(jets[i].p4.p()).Normalize();
			h_OnAxis.push_back(jets[i].OnAxis(lMax));
			h_OnAxis.back() *= jets[i].p4.x0;
			
			for(size_t j = i + 1; j < jets.size(); ++j)
				Pl_computer.z[k++] = pHat_i.Dot(vec3_t(jets[j].p4.p()).Normalize());
		}
		
		assert(k <= incrementSize);
	}
	
	Pl_computer.Reset();
	incrementArray_t 
		weight,
		H_l_accumulate; // the elements (P_l * fProd)_k, which we sum
		
	weight.fill(real_t(0));
			
	for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
	{
		H_l_accumulate.fill(real_t(0));
		auto const& P_l_increment = Pl_computer.Next(); // Get the next P_l
		
		for(auto const& h : h_OnAxis)
			H_l_vec[lMinus1] += kdp::Squared(h[lMinus1 + 1]);
			
		{
			size_t k = 0;
		
			for(size_t i = 0; i < jets.size(); ++i)
			{
				for(size_t j = i + 1; j < jets.size(); ++j)
					weight[k++] = real_t(2) * h_OnAxis[i][lMinus1 + 1] * h_OnAxis[j][lMinus1 + 1];
			}
		}
		
		for(size_t k = 0; k < incrementSize; ++k)
			H_l_accumulate[k] = P_l_increment[k] * weight[k];
		
		H_l_vec[lMinus1] += kdp::BinaryAccumulate_Destructive(H_l_accumulate);
	}
	
	assert(detectorFilter.size() > lMax);
	for(size_t i = 0; i < lMax; ++i)
		H_l_vec[i] *= detectorFilter[i + 1];		
	
	return H_l_vec;
}

// WARNING: something broken here
std::vector<SpectralPower::real_t> SpectralPower::Power_Jets_Particles(size_t const lMax,
	std::vector<ShapedJet> const& jets,
	std::vector<PhatF> const& tracks, std::vector<real_t> const& h_OnAxis_tracks, 
	std::vector<PhatF> const& towers, std::vector<real_t> const& h_OnAxis_towers)
{
	auto H_l_vec = Power_Jets_Particles_SubTerm(lMax, jets, tracks, h_OnAxis_tracks);
	H_l_vec += Power_Jets_Particles_SubTerm(lMax, jets, towers, h_OnAxis_towers);
	
	return H_l_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Power_Jets_Particles_SubTerm(size_t const lMax,
	std::vector<ShapedJet> const& jets,
	std::vector<PhatF> const& particles, std::vector<real_t> const& h_OnAxis_particles)
{
		// Allocate a vector of H_l to return, which we constantly add to (so initialize to zero).
	// H_l[0] actually corresponds to l == 1 (because H_0 = 1 always), so allocate lMax.
	std::vector<real_t> H_l_vec(lMax, real_t(0));
	size_t constexpr incrementSize = 32;
	assert(H_l_vec.size() == lMax);
	
	if(H_l_vec.size())
	{
		assert((h_OnAxis_particles.size() - 1) >= lMax);
		
		RecursiveLegendre<real_t, incrementSize> Pl_computer;
		
		using incrementArray_t = decltype(Pl_computer)::incrementArray_t;	
		
		incrementArray_t 
			fProd_increment, // increments of fProd, filled by outer
			H_l_accumulate; // the elements (P_l * fProd)_k, which we sum
		
		for(ShapedJet const& jet : jets)
		{
			auto const lFactor = [&](){
				auto h_OnAxis_jet = jet.OnAxis(lMax);
				
				// Assume h_OnAxis starts with l=0, which we don't want
				assert(h_OnAxis_jet.front() == real_t(1));
				assert(h_OnAxis_particles.front() == real_t(1));
				
				h_OnAxis_jet.erase(h_OnAxis_jet.begin()); // We don't need the leading 1
				for(size_t l = 0; l < lMax; ++l)
					h_OnAxis_jet[l] *= h_OnAxis_particles[l + 1];
					
				return h_OnAxis_jet;}();
			assert(lFactor.size() == lMax);
				
			vec3_t const jet_pHat = vec3_t(jet.p4.p()).Normalize();
			real_t const jet_f = jet.p4.x0;
				
			size_t k = 0;
			
			for(size_t j = 0; j < particles.size(); ++j)
			{
				fProd_increment[k] = jet_f * particles[j].f;
				Pl_computer.z[k] = jet_pHat.Dot(particles[j].pHat);
				++k;
				
				// Fill the final increment with zeros
				if(j == (particles.size() - 1))
				{
					while(k < incrementSize)
					{
						fProd_increment[k] = real_t(0);
						Pl_computer.z[k] = real_t(0);
						++k;
					}
				}
				
				if(k == incrementSize)
				{
					Pl_computer.Reset();
					//~ assert(Pl_computer.l() == 0);
				
					for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
					{
						//~ std::cout << lMinus1 << "\t" << size_t(H_l_vec.data()) << std::endl;
						auto const& P_l_increment = Pl_computer.Next(); // Get the next P_l
						//~ assert((lMinus1 + 1) == Pl_computer.l());
						
						// Do a dot product with fProd
						for(size_t m = 0; m <incrementSize; ++m)
							H_l_accumulate[m] = P_l_increment[m] * fProd_increment[m];
									
						H_l_vec[lMinus1] += lFactor[lMinus1] * 
							kdp::BinaryAccumulate_Destructive(H_l_accumulate);
					}// Done l-loop
					
					k = 0; // MUST reset k 
				}
				
				assert(k < incrementSize);
			}
		}
	}
	
	return H_l_vec;
}

