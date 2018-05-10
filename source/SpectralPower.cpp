#include "SpectralPower.hpp"
#include "SelfOuterU.hpp"
#include "NjetModel.hpp"
#include "kdp/kdpTools.hpp"
#include "RecursiveLegendre.hpp"
#include <array>
#include <future>

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

//~ std::vector<SpectralPower::PhatF> SpectralPower::PhatF::To_PhatF_Vec
	//~ (std::vector<Pythia8::Particle> const& original)
//~ {
	//~ double totalE = 0.;
	
	//~ std::vector<PhatF> newVec;
	
	//~ for(Pythia8::Particle const& particle : original)
	//~ {
		//~ newVec.emplace_back(particle);
		//~ totalE += particle.e();
	//~ }
		
	//~ for(auto& pHatF : newVec)
		//~ pHatF.f /= totalE;
		
	//~ return newVec;
//~ }

////////////////////////////////////////////////////////////////////////

std::vector<SpectralPower::PhatF> SpectralPower::PhatF::To_PhatF_Vec
	(std::vector<vec3_t> const& originalVec)
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

////////////////////////////////////////////////////////////////////////

SpectralPower::PhatFvec::PhatFvec(std::vector<PhatF> const& orig)
{
	reserve(orig.size());
	for(PhatF const& p : orig)
	{
		this->emplace_back(p);
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

void SpectralPower::PhatFvec::emplace_back(PhatF const& pHatF)
{
	// Not the most efficient way to normalize, but not the killer operation
	x.emplace_back(pHatF.pHat.x1);
	y.emplace_back(pHatF.pHat.x2);
	z.emplace_back(pHatF.pHat.x3);
	
	f.emplace_back(pHatF.f);
}

////////////////////////////////////////////////////////////////////////

//~ void SpectralPower::PhatFvec::emplace_back
	//~ (vec3_t const& pHat, real_t const f_in, bool const normalize)
//~ {
	//~ // Not the most efficient way to normalize, but not the killer operation
	//~ real_t const norm = (normalize ? pHat.Mag() : 1.);
	//~ x.emplace_back(pHat.x1/norm);
	//~ y.emplace_back(pHat.x2/norm);
	//~ z.emplace_back(pHat.x3/norm);
	
	//~ f.emplace_back(f_in);
//~ }

////////////////////////////////////////////////////////////////////////

SpectralPower::PhatFvec SpectralPower::PhatFvec::Join
	(PhatFvec&& first, PhatFvec&& second)
{
	// Steal the data in first, using the move ctor
	//~ PhatFvec joined(std::move(first));
	
	// Append second
	first.x.insert(first.x.end(), second.x.begin(), second.x.end());
	first.y.insert(first.y.end(), second.y.begin(), second.y.end());
	first.z.insert(first.z.end(), second.z.begin(), second.z.end());
	
	first.f.insert(first.f.end(), second.f.begin(), second.f.end());
	
	return first;
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

std::vector<SpectralPower::real_t> SpectralPower::Hl(size_t const lMax, 
	PhatFvec const& particles, size_t const numThreads_requested)
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
	
	std::vector<real_t> Hl_vec; // We return a vector of H_l. 
	
	if(numThreads <= 1) // if numThreads == 0 (numIncrements  < minIncrements), use 1 thread.
		Hl_vec = H_l_threadedIncrement();
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
			std::vector<real_t> Hl_vec_thread = threadReturn[t].get();
			
			if(t == 0) // Intialize to first increment by stealing thread's data
				Hl_vec = std::move(Hl_vec_thread);
			else
			{
				// Go through each H_l returned and add it to the running sum.
				// The number of threads is limited, so there's no point in a binary sum.
				for(size_t lMinus1 = 0; lMinus1 < settings.lMax; ++lMinus1)
					Hl_vec[lMinus1] += Hl_vec_thread[lMinus1];
			}
		}
	}
		
	if(settings.lFactor) // Add (2l + 1) prefix
	{
		// index 0 contains l=1, so the prefix is 2*(lMinus1+1)+1 = 2*lMinus1 + 3
		for(size_t lMinus1 = 0; lMinus1 < settings.lMax; ++lMinus1)
			Hl_vec[lMinus1] *= real_t(2*lMinus1 + 3);
	}
	
	if(settings.nFactor) // Divide by <f|f>, to normalize for multiplicity
	{
		real_t f2sum;
		
		// Calculate f2sum using a binary reduction. This limits floating-point error
		// with the same algorithm that Hl_vec used
		{
			std::vector<real_t> fCopy = particles.f;
			for(real_t& f : fCopy)
				f = f*f;
				
			f2sum = kdp::BinaryAccumulate_Destructive(fCopy);
		}
		
		for(real_t& H_l : Hl_vec)
			H_l /= f2sum;
	}
	
	return Hl_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Hl(size_t const lMax, 
	std::vector<vec3_t> const& particles, size_t const numThreads_requested)
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
	
	return Hl(lMax, PhatF::To_PhatF_Vec(particles), numThreads_requested);
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
		auto const H = Hl(lMax, particles, numThreads_requested);
		
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
	std::vector<real_t> Hl_vec(settings.lMax, 0.);
	
	if(Hl_vec.size())
	{
		RecursiveLegendre_Increment<std::array<real_t, Outer_Increment::incrementSize>> Pl_computer;
		
		using container_t = decltype(Pl_computer)::container_t;	
		
		container_t 
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
				
			// We start Pl_computer at l=1, but we don't want to call Pl_computer.Next()
			// too many times. If we call Next() at the start of each iteration, 
			// then we need to access l-1 inside the loop
			for(real_t& H_l : Hl_vec) // Recursively set H_l
			{
				//~ auto const P_l_increment = Pl_computer.Next(); // Get the next P_l
				
				// Do a dot product with fProd
				for(size_t i = 0; i < Outer_Increment::incrementSize; ++i)
					H_l_accumulate[i] = Pl_computer.P_lm1()[i] * fProd_increment[i];
							
				H_l += kdp::BinaryAccumulate_Destructive(H_l_accumulate);
			}// Done l-loop
		}// Done increment loop
	}// Done with outer
	
	return Hl_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Hl_Obs(size_t const lMax,
	std::vector<PhatF> const& particles, ShapeFunction const& particleShape)
{
	return Hl_Extensive_SelfTerm(lMax, particles, particleShape.OnAxis(lMax));
}

std::vector<SpectralPower::real_t> SpectralPower::Hl_Obs(size_t const lMax,
	std::vector<PhatF> const& tracks, ShapeFunction const& trackShape, 
	std::vector<PhatF> const& towers, ShapeFunction const& towerShape)
{
	auto Hl_vec = Hl_Extensive_SelfTerm(lMax, tracks, trackShape.OnAxis(lMax));
	Hl_vec += Hl_Extensive_SelfTerm(lMax, towers, towerShape.OnAxis(lMax));
	Hl_vec += Hl_Extensive_SubTerm(lMax, 
		tracks, trackShape.OnAxis(lMax), 
		towers, towerShape.OnAxis(lMax))*real_t(2);
	
	return Hl_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Hl_Extensive_SelfTerm(size_t const lMax,
	std::vector<PhatF> const& particles, std::vector<real_t> const& hl_OnAxis)
{
	// Semi-redundant calculation of self terms, but good enough for now.
	return Hl_Extensive_SubTerm(lMax, particles, hl_OnAxis, particles, hl_OnAxis);
}

// We need this function for asymmetric outer products
std::vector<SpectralPower::real_t> SpectralPower::Hl_Extensive_SubTerm(size_t const lMax,
	std::vector<PhatF> const& left, std::vector<real_t> const& hl_OnAxis_left, 
	std::vector<PhatF> const& right, std::vector<real_t> const& hl_OnAxis_right)
{
	// Allocate a vector of Hl to return, which we constantly add to (so initialize to zero).
	// Hl_vec[0] actually corresponds to l == 1 (because H_0 = 1 always), so allocate lMax.
	std::vector<real_t> Hl_vec(lMax, 0);
	size_t constexpr incrementSize = 64;
	
	if(Hl_vec.size())
	{
		// Because the first hl are always one (we should really fix that)
		assert(hl_OnAxis_left.size() >= lMax);
		assert(hl_OnAxis_right.size() >= lMax);
			
		using incrementArray_t = std::array<real_t, incrementSize>;
			
		RecursiveLegendre_Increment<incrementArray_t> Pl_computer;
	
		incrementArray_t 
			fProd_increment, // increments of fProd, filled by outer
			H_l_accumulate; // the elements (P_l * fProd)_k, which we sum
		
		size_t k = 0;
		
		// inefficient but correct
		// REMEMBER: when these vectors are not the same, the matrix is not symmetric
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
					assert(Pl_computer.l() == 1);
				
					for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
					{
						//~ auto const& P_l_increment = Pl_computer.Next(); // Get the next P_l
						//~ assert((lMinus1 + 1) == Pl_computer.l());
						Pl_computer.Next();
						
						// Do a dot product with fProd
						for(size_t m = 0; m <incrementSize; ++m)
							H_l_accumulate[m] = Pl_computer.P_lm1()[m] * fProd_increment[m];
									
						Hl_vec[lMinus1] += kdp::BinaryAccumulate_Destructive(H_l_accumulate);
					}// Done l-loop
					
					k = 0; // Start the next increment
				}
			}
		}
		
		for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
			Hl_vec[lMinus1] *= (hl_OnAxis_left[lMinus1] * hl_OnAxis_right[lMinus1]);
	}
	
	return Hl_vec;
}

std::vector<SpectralPower::real_t> SpectralPower::Hl_Jet(size_t const lMax,
	std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter)
{
	std::vector<real_t> Hl_vec(lMax, 0);
	assert(hl_onAxis_Filter.size() >= lMax);
		
	RecursiveLegendre_Increment<std::vector<real_t>> Pl_computer;
	
	std::vector<std::vector<real_t>> hl_i_onAxis;
	
	{
		std::vector<vec3_t> pHat;
		pHat.reserve(jets.size());
				
		for(size_t i = 0; i < jets.size(); ++i)
		{
			pHat.push_back(vec3_t(jets[i].p4.p()).Normalize());
			
			// Push back each jet's on-axis coefficients, weighted by the jet's f 
			auto const& hl_i = jets[i].OnAxis(lMax);
			
			hl_i_onAxis.emplace_back(hl_i.begin(), hl_i.begin() + lMax);
			hl_i_onAxis.back() *= jets[i].p4.x0;
			
			for(size_t j = 0; j < i; ++j)
				Pl_computer.z.push_back(pHat[i].Dot(pHat[j]));
		}
	}
		
	Pl_computer.Reset();
	size_t const numTerms = Pl_computer.z.size();
	
	std::vector<real_t> weight, // weight the Pl for each l 
		Hl_accumulate; // the elements (P_l * fProd)_k, which we sum
	
	for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
	{
		Pl_computer.Next(); // Get the next P_l
		weight.assign(numTerms, real_t(0));
		Hl_accumulate.assign(numTerms, real_t(0));
		
		// Add the self-correlation
		for(auto const& h : hl_i_onAxis)
			Hl_vec[lMinus1] += kdp::Squared(h[lMinus1]);
		
		{
			size_t k = 0;
			
			for(size_t i = 0; i < jets.size(); ++i)
			{
				for(size_t j = 0; j < i; ++j)
					weight[k++] = real_t(2) * hl_i_onAxis[i][lMinus1] * hl_i_onAxis[j][lMinus1];
			}
		}
		
		for(size_t k = 0; k < numTerms; ++k)
			Hl_accumulate[k] = Pl_computer.P_lm1()[k] * weight[k];
		
		Hl_vec[lMinus1] += kdp::BinaryAccumulate_Destructive(Hl_accumulate);
	}
	
	for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
		Hl_vec[lMinus1] *= kdp::Squared(hl_onAxis_Filter[lMinus1]);
	
	return Hl_vec;
}

// WARNING: something broken here
std::vector<SpectralPower::real_t> SpectralPower::Hl_Hybrid(size_t const lMax,
	std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter,
	std::vector<PhatF> const& tracks, ShapeFunction const& trackShape, 
	std::vector<PhatF> const& towers, ShapeFunction const& towerShape, 
	std::vector<real_t> const& Hl_Obs_in)
{
	std::vector<real_t> Hl_vec;
	
	if(Hl_Obs_in.empty())
		Hl_vec = SpectralPower::Hl_Obs(lMax, tracks, trackShape, towers, towerShape); 
	else
		Hl_vec.assign(Hl_Obs_in.begin(), Hl_Obs_in.begin() + lMax);
		
	assert(Hl_vec.size() == lMax);
	
	Hl_vec += Hl_Jet(lMax, jets, hl_onAxis_Filter);
	
	Hl_vec += Hl_Jets_Particles_SubTerm(lMax, 
		jets, hl_onAxis_Filter, 
		tracks, trackShape.OnAxis(lMax))*real_t(2);
		
	Hl_vec += Hl_Jets_Particles_SubTerm(lMax, 
		jets, hl_onAxis_Filter, 
		towers, towerShape.OnAxis(lMax))*real_t(2);
		
	return Hl_vec * 0.25;
}

std::vector<SpectralPower::real_t> SpectralPower::Hl_Jets_Particles_SubTerm(size_t const lMax,
	std::vector<ShapedJet> const& jets, std::vector<real_t> const& hl_onAxis_Filter,
	std::vector<PhatF> const& particles, std::vector<real_t> const& hl_onAxis_particles)
{
	// Allocate a vector of H_l to return, which we constantly add to (so initialize to zero).
	// H_l[0] actually corresponds to l == 1 (because H_0 = 1 always), so allocate lMax.
	std::vector<real_t> Hl_vec(lMax, real_t(0));
	size_t constexpr incrementSize = 16;
		
	if(Hl_vec.size())
	{
		assert(hl_onAxis_Filter.size() >= lMax);
		assert(hl_onAxis_particles.size() >= lMax);
		
		using incrementArray_t = std::array<real_t, incrementSize>;
		
		RecursiveLegendre_Increment<incrementArray_t> Pl_computer;
				
		incrementArray_t 
			fProd_increment, // increments of fProd, filled by outer
			Hl_accumulate; // the elements (P_l * fProd)_k, which we sum
		
		for(ShapedJet const& jet : jets)
		{
			auto const& hl_OnAxis_jet = jet.OnAxis(lMax);
			vec3_t const jet_pHat = vec3_t(jet.p4.p()).Normalize();
							
			size_t k = 0;
			
			for(size_t j = 0; j < particles.size(); ++j)
			{
				fProd_increment[k] =  particles[j].f;
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
									
					for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
					{
						Pl_computer.Next(); // Get the next P_l
												
						// Do a dot product with fProd
						for(size_t m = 0; m < incrementSize; ++m)
							Hl_accumulate[m] = Pl_computer.P_lm1()[m] * fProd_increment[m];
									
						Hl_vec[lMinus1] += hl_OnAxis_jet[lMinus1] * jet.p4.x0 * 
							kdp::BinaryAccumulate_Destructive(Hl_accumulate);
					}// Done l-loop
					
					k = 0; // Reset k to fill next increment
				}
				
				assert(k < incrementSize);
			}
		}
		
		for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
			Hl_vec[lMinus1] *= hl_onAxis_Filter[lMinus1] * hl_onAxis_particles[lMinus1];
	}
	
	return Hl_vec;
}
