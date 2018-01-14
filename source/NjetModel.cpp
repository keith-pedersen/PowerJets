#include "NjetModel.hpp"
#include "RecursiveLegendre.hpp"
#include "pqRand/pqRand.hpp"
#include <algorithm> // std::max
#include "helperTools.hpp"

void Jet::SampleShape(incrementArray_t& z, incrementArray_t& xy_sinPhi, 
	pqRand::engine& gen) const
{
	// We will boost z_CM into the lab frame
	// We assume boost collimates particles towards +z axis in the lab
	real_t const beta = p4.p().Mag() / p4.x0;
	static constexpr size_t subIncrement = (incrementSize / 2);
	
	for(size_t i = 0; i < subIncrement; ++i)
	{
		// Draw one u to generate antithetic z values (hopefully to reduce variance)
		real_t const u = gen.U_uneven();
		// NOTE: must draw independent phi to avoid correlation between z and phi
		
		// z = z_lab =  (beta + z_CM)/(1 + beta * z_CM)	
		{
			// w+ = 1 - z = (1 - beta)(1 - z_CM)/(1 + beta * z_CM)
			real_t const w_plus = u / ((real_t(1) + beta) * kdp::Squared(p4.x0 / mass) * 
				(real_t(1) + beta * (real_t(1) - u)));
			//~ assert(w_plus > real_t(0));
			//~ assert(w_plus <= real_t(2));
			
			z[i] = real_t(1) - w_plus;
			xy_sinPhi[i] = std::sqrt(w_plus * (real_t(2) - w_plus)) * 
				gen.ApplyRandomSign(std::sin(gen.U_uneven() * M_PI_2));
		}
		
		{
			// w- = 1 + z 
			// this is the less accurate one
			real_t const w_minus = u*(real_t(1) + beta)/(real_t(1) + beta*(u - real_t(1)));
			//~ assert(w_minus > real_t(0));
			//~ assert(w_minus <= real_t(2));
			
			z[subIncrement + i] = w_minus - real_t(1);
			xy_sinPhi[subIncrement + i] = std::sqrt(w_minus * (real_t(2) - w_minus)) * 
				gen.ApplyRandomSign(std::sin(gen.U_uneven() * M_PI_2));
		}
	}
}

NjetModel::NjetModel(QSettings const& settings):
	gen() {} //, computer(settings), detector(ArrogantDetector::NewDetector(settings)) {}
	
NjetModel::NjetModel(std::string const& iniFileName):
	NjetModel(QSettings(iniFileName.c_str(), QSettings::IniFormat)) {}

NjetModel::NjetModel():NjetModel(QSettings()) {}

NjetModel::~NjetModel() {}//delete detector;}

std::vector<NjetModel::real_t> NjetModel::operator()
	(std::vector<Jet> const& jetVec_unsorted, 
	size_t const lMax, real_t const jetShapeGranularity) const
{
	// Return H_l from (l = 1) to (l = lMax)
	std::vector<real_t> H_l_vec(lMax, 0.);
	
	/* Defining the integral
	 * 	rho_(i) = f_i * f(z) / (2 pi)
	 * 	rho_(i)_l = \int P_l(z) rho_(i) dz dphi = f_i \int P_l(z) f(z) dz
	 * We need to calculate
	 * 	H_l = Etot**(-2)*((rho_(1)_l)**2 + rho_(1)_l * rho_(2)_l + ... + rho_(2)_l * rho_(1)_l + ... )
	 * However the integral is difficult. Instead, we use 
	 * Monte Carlo integration for n variates drawn from f(z)
	 * 	\rho_(i)_l ~= 1/n * sum_k P_l(z_k) * f(z_k) / f(z_k) = 1/n * sum_k P_l(z_k)
	 * We can compute each cross-term {rho_(i)_l * rho_(j)_l} as 
	 * {rho_(i,j)_l * rho_(j)_l}
	 * 	a) compute rho_(j)_l with jet_j parallel to the z-axis,
	 * 		and merely boosted from its CM frame to the lab frame.
	 *		b) compute rho_(i,j)_l with jet_i starting parallel to the z-axis, 
	 * 		boosted into the lab frame along the z-axis, 
	 * 		then rotated off axis by the ij interior angle.
	 * This naively requires N = (n_jets * n)**2 / 2 variates and Order(N * lMax) FLOPS.
	 * However, this is quite redundant, as we can reuse variates.
	 * 
	 * 1. For each jet_i, draw n variates.
	 * 2. For j < i, jet_j is the jet parallel to the z-axis, 
	 * 	and jet_i is the rotated jet. Compute rho_(i,j)_l and 
	 * 	multiply by the pre-computed rho_(j)_l (which only needs to be 
	 * 	computed once, since it does not depend on any angle).
	 * 3. When j == i, compute rho_(j)_l for use by all subsequent i.
	 * 
	 * Since n is proportional to energy (because smaller jets are less important),
	 * sorting jets from largest energy to smallest minimizes the total 
	 * number of FLOPS (because we need Order(n * lMax) FLOPS per cross-term, 
	 * and there is one additional cross-term each time i increments).
	*/ 
	 
	if(lMax > 0)
	{
		// First sort the incoming jetVec from high to low energy.
		std::vector<Jet> jetVec = jetVec_unsorted;
		
		// We can use lambdas (no-capture, empty square bracket) to use some standard tools with Jet class
		std::sort(jetVec.begin(), jetVec.end(), 
			[](Jet const& lhs, Jet const& rhs) {return lhs.p4.x0 > rhs.p4.x0;});
		
		// Add up energy from back to front (small to large)
		real_t const Etot = std::accumulate(jetVec.crbegin(), jetVec.crend(), real_t(0), 
			// a lambda which sums jet energy for a std::accumulate
			[](real_t const E_current, Jet const& jet) {return E_current + jet.p4.x0;});
			
		using incrementArray_t = Jet::incrementArray_t;
		
		// Each jet's will fill shape variate positions into these two arrays
		incrementArray_t z_lab, y_lab;
		
		std::vector<std::vector<real_t>> rho; // The rho for each jet_j (and each l)
		RecursiveLegendre<real_t, Jet::incrementSize> Pl_computer;
		
		// In the following outer loops, we will use i and j because it makes
		// the code more readable. Being outer loops, the penalty is tiny.
		for(size_t i = 0; i < jetVec.size(); ++i)
		{
			Jet const& jet_i = jetVec[i];
			
			size_t const n_requested = 
				std::max(1lu, 
					//~ (jet_i->p4.x0 < 1e3*jet_i->mass) ? 0 : 
						size_t(jet_i.p4.x0 * jetShapeGranularity / Etot));
			size_t const n_increments = MinPartitions(n_requested, Jet::incrementSize);
			
			// The number of variates can be very high, so we obtain
			// them in increments, and for each increment loop over jet_j.
			// This minimizes the memory overhead.
			// To avoid the redundant calculation of inter-jet cos/sin as we 
			// repeatedly loop over the same j, we calculate then during
			// the first increment and reuse it for all others.
			std::vector<std::pair<real_t, real_t>> cos_sin_ij;
			cos_sin_ij.reserve(i + 1);
			
			// We accumulate each jet_i's rho before placing it in H_l,
			// because each jet_i has its own normalization, 
			// and we would prefer to normalize as little as possible.
			// Also we can accumulate all of rho_i before
			// adding it to the sum over all rho_i*rho_j.
			std::vector<std::vector<real_t>> rho_accumulate(i + 1, 
				std::vector<real_t>(lMax, 0));
						
			for(size_t k = 0; k < n_increments; ++k)
			{
				// These are the positions when jet_i is boosted to the lab, 
				// but still parallel to the z axis. They must be rotated off-axis
				// for each jet_j.		
				jet_i.SampleShape(z_lab, y_lab, gen);
				
				for(size_t j = 0; j <= i; ++j)
				{
					if(k == 0) // First increment, find and cache interior angle
					{
						if(j == i)
							cos_sin_ij.emplace_back(1, 0);
						else
						{
							vec3_t const& p3_i = jet_i.p4.p();
							vec3_t const& p3_j = jetVec[j].p4.p();
										
							real_t const mag2_ij = p3_i.Mag2() * p3_j.Mag2();
							
							cos_sin_ij.emplace_back(
								p3_i.Dot(p3_j)/std::sqrt(mag2_ij), 
								std::sqrt(p3_i.Cross(p3_j).Mag2()/mag2_ij));
						
							// Check for over-unity from rounding (but they can't BOTH be over-unity)
							// Rare, so who cares that round is slower than copysign
							if(std::abs(cos_sin_ij.back().first) > real_t(1))
								cos_sin_ij.back().first = std::round(cos_sin_ij.back().first);
							else if(std::abs(cos_sin_ij.back().second) > real_t(1))
								cos_sin_ij.back().second = std::round(cos_sin_ij.back().second);
						}
					}
						
					auto const& cos_sin = cos_sin_ij[j];
					
					for(size_t m = 0; m < Jet::incrementSize; ++m)
					{
						Pl_computer.z[m] = cos_sin.first * z_lab[m] + cos_sin.second * y_lab[m];
						
						//~ // When the CM of the jets is boosted a bit in the lab frame, 
						//~ // I was encountering a problem where
						//~ // H_l would diverge quite significantly after only a few l.
						//~ // Installing this bounds check completely fixed the problem.
						//~ if(std::fabs(Pl_computer.z[m]) > real_t(1))
						//~ {
							//~ printf("(%.3e %.3e) (%.3e, %.3e) %.16f\n", cos_sin.first, cos_sin.second, 
								//~ z[m], xy_sinPhi[m],
								//~ Pl_computer.z[m]);
							//~ Pl_computer.z[m] = std::round(Pl_computer.z[m]);
						//~ }
					}
					
					Pl_computer.Reset();
					
					// loop over l
					for(auto& rho_accumulate_j_l : rho_accumulate[j]) 
						rho_accumulate_j_l += BinaryAccumulate(Pl_computer.Next());
				}
			}
			
			// Do jet_i's energy and sample-size normalization once (to reduce FLOPS and rounding error)
			{
				real_t const normalization = jet_i.p4.x0 / real_t(n_increments * Jet::incrementSize);
				
				for(size_t j = 0; j < i; ++j)
				{
					auto const& rho_j = rho[j];
					auto& rho_accumulate_j = rho_accumulate[j];
					
					for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
					{
						H_l_vec[lMinus1] += real_t(2) * 
							rho_j[lMinus1] * (normalization * rho_accumulate_j[lMinus1]);
					}
				}
				
				// Emplace the rho for jet_i (the self rho)
				rho.emplace_back(rho_accumulate.back());
				auto& rho_i = rho.back();
					
				for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
					H_l_vec[lMinus1] += kdp::Squared(rho_i[lMinus1] *= normalization);
			}
			
			//~ if(i == 1)
			//~ {
				//~ std::ofstream file("rho_self.dat", std::ios::trunc);
				//~ char buff[1024];
								
				//~ sprintf(buff, "# %.5e\n", jet_i.p4.x0/jet_i.mass);
				//~ file << buff;
				
				//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
				//~ {
					//~ sprintf(buff, "%lu  %.5e  %.5e\n", lMinus1 + 1, rho[0][lMinus1]/jetVec[0].p4.x0, rho[1][lMinus1]/jetVec[1].p4.x0);
					//~ file << buff;
				//~ }
			//~ }
		}
		
		// Do the energy normalization once (to reduce FLOPS and rounding error)
		{
			real_t const normalization = kdp::Squared(Etot);
		
			for(real_t& H_l : H_l_vec)
				H_l /= normalization;
		}
	}
		
	return H_l_vec;
	
	// Original, particle Monte Carlo
	
	//~ std::vector<vec4_t> particles;
	
	//~ vec4_t const total = std::accumulate(jetVec.begin() + 1, jetVec.end(), jetVec.front());
		
	//~ for(vec4_t const& jet : jetVec)
	//~ {
		//~ double const mass = jet.Length();
		
		//~ if(mass <= 0.) // Rely on relative difference
			//~ particles.push_back(jet);
		//~ else
		//~ {
			//~ kdp::LorentzBoost<real_t> const boost(jet);
			
			//~ size_t nIsoRequested = std::max(size_t(3),
				//~ size_t(std::min(1., std::ceil(mass / total.x0)) * jetShapeGranularity));
			
			//~ // nCopies = f_jet * jetShapeGranularity / boost = E / totalE * gran / (E / m) = 
			//~ std::vector<vec3_t> isoVec = IsoCM(nIsoRequested, gen);
				
			//~ // Split the mass evenly between the iso vecs 
			//~ // (note that we may get one more than nIsoRequested)
			//~ double const E_CM = mass / double(isoVec.size());
			
			//~ for(vec3_t& iso : isoVec)
			//~ {
				//~ assert(std::fabs(iso.Mag() - 1.) < 1e-14);
				
				//~ iso *= 0.9 * E_CM; // Scale the particle to the correct energy
				
				//~ // Create a massless 4-vector
				//~ kdp::Vec4 const p4_iso(E_CM, iso.x1, iso.x2, iso.x3);

				//~ particles.push_back(boost(p4_iso)); // Boost into the lab frame
				//~ if(not(particles.back().Length2() == particles.back().Length2()))
				//~ {
					//~ printf("error: %2e\n", mass);
				//~ }
			//~ }
		//~ }
	//~ }
	
	//~ // (*detector)(particles); // Send n-jet model as neutral
	//~ // printf("(%lu, %lu)\n", particles.size(), detector->Towers().size());
	//~ // return computer(detector->Towers(), lMax); // Return the spectral power
	
	//~ SpectralPower::PhatFvec t;
	
	//~ for(kdp::Vec4 const& part : particles)
		//~ t.emplace_back(part.p()/part.x0, part.x0/total.x0, false);
		
	//~ return computer(t, lMax); // Return the spectral power
}
	
/*


std::vector<NjetModel::vec4_t> NjetModel::GetJets(std::vector<double> const& jetParams_in)
{
	std::vector<vec4_t> jets;		
	{
		std::vector<double> jetParams = jetParams_in; // Copy so we can alter
		
		// Every jet has 4 parameters {p_ix, p_iy, p_iz, m_i} concatenated into jetParams.
		// Originally, p_i was interpreted as the actual momentum, so that E_i**2 = p_i**2 + m_i**2.
		// However, the fit was having a hard time, and the hypothesis was that
		// changing the mass now alters H_l too wildly. 
		// The mass parameter should spread the energy around, 
		// but it should not also change the energy fraction, and thus the fundamental nature of H_l.
		// The second attempt interprets p_i as the momentum of the massless jet, 
		// so that changing the mass reduces the physical momentum but does not 
		// alter the jet's energy fraction.
		// However, in either case the energy fraction of the final jet is altered by 
		// changing the mass of any given jet, so m2OverP2_balancing is the same as before.
		
		// After all jets are created a final jet is added to place the model in the CM frame.
		// If there are one-too-many parameters in jetParams (4*n + 1), 
		// it is interpreted as  the boost of the balancing jet.
		double const m2OverP2_balancing = ((jetParams.size() % 4) == 1) ? 
			1./Diff2(jetParams.back(), 1.) : 0.;
			//Solve[Sqrt[1 + m2]/Sqrt[m2] == \[Gamma], m2]
		
		if((jetParams.size() % 4) > 1)
			throw std::runtime_error("NjetModel: the number of parameters must be 4n or 4n + 1");
		
		// We will place the balancing jet in the final param spot,
		// to reduce the redundancy in the jet creation loop
		jetParams.insert(jetParams.end(), 4 - (jetParams.size() % 4), 0.);
		assert((jetParams.size() % 4) == 0);
			
		// Convert fitting parameters to jets
		for(size_t i = 0; i < jetParams.size(); i += 4)
		{	
			vec3_t p3(jetParams[i], jetParams[i + 1], jetParams[i + 2]);
			double const mass = jetParams[i + 3];
			
			double const energy = std::sqrt(p3.Mag2() + kdp::Squared(mass));
			//~ double const energy = p3.Mag(); // v2
			
			assert(energy >= mass);
			
			if(energy > (1e3 * mass)) // If the boost is too high
				jets.emplace_back(p3); // The jet has no extent, create it as massless.
			else
			{
				//~ p3 *= std::sqrt(Diff2(1., mass/energy)); //v2
				jets.emplace_back(energy, p3, kdp::Vec4from2::Energy);
			}
			
			if(i + 8 == jetParams.size())
			{
				// The next iteration will be final param spot, which is the balancing jet.
				// We will now emplace the balancing p3 and mass.
								
				vec4_t const total = std::accumulate(jets.begin() + 1, jets.end(), jets.front());
				
				jetParams[i + 4] = -(total.p().x1);
				jetParams[i + 5] = -(total.p().x2);
				jetParams[i + 6] = -(total.p().x3);
				jetParams[i + 7] = std::sqrt(m2OverP2_balancing * total.p().Mag2());
			}
		}
		
		if(jets.back().x0 == 0.) jets.pop_back();
	}
	
	return jets;	
}

//~ std::vector<std::vector<NjetModel::real_t>> NjetModel::GetJetsPy(std::vector<double> const& jetParams)
//~ {
	//~ std::vector<std::vector<real_t>> jetRet;
	//~ {
		//~ auto const jetVec = GetJets(jetParams);
		//~ jetRet.assign(jetVec.begin(), jetVec.end()); // Assign will auto-convert to std::vector
	//~ }	
	//~ return jetRet;
//~ }

////////////////////////////////////////////////////////////////////////

SpectralPower::vec3_t NjetModel::IsoVec3(pqRand::engine& gen)
{
	// The easiest way to draw an isotropic 3-vector is rejection sampling.
	// It provides slightly better precision than a theta-phi implementation,
	// and is not much slower.
	
	using vec3_t = SpectralPower::vec3_t;
	
	vec3_t iso(false); // false means don't initialize the Vec3
	double r2;
	
	do
	{
		// U_S has enough precision for this application
		// because we scale by r2
		iso.x1 = gen.U_even();
		iso.x2 = gen.U_even();
		iso.x3 = gen.U_even();
		
		r2 = iso.Mag2();
	}
	while(r2 > 1.); // Draw only from the unit sphere (in the positive octant)
	
	// It we wanted to scale the length from some distribution, 
	// the current length r**2 is a random variate (an extra DOF since 
	// only two DOF are required for isotropy) which can be 
	// easily converted into a random U(0, 1) via its CDF 
	// 	u = CDF(r2) = (r2)**(3/2)
	// (you can work this out from the differential volume dV/(V * d(r**2))).
	// This U(0,1) can then be plugged into any quantile function Q(u)
	// for the new length. One should check to see if 
	// dividing out by the current length can be rolled into Q(u), 
	// so that you that you use u to draw the scale that the takes 
	// iso to its new length.
	
	// In this application, we simply want unit vectors, 
	// so we throw away the random entropy of r2.
	iso /= std::sqrt(r2);
		
	// Use random signs to move to the other 7 octants
	gen.ApplyRandomSign(iso.x1);
	gen.ApplyRandomSign(iso.x2);
	gen.ApplyRandomSign(iso.x3);
	
	return iso;
}

////////////////////////////////////////////////////////////////////////

// Return a vector of random 3-vectors isotropically distributed, 
// but which nonetheless sum to zero. 
std::vector<SpectralPower::vec3_t> NjetModel::IsoCM(size_t const n_request, pqRand::engine& gen,
	double const tolerance)
{
	using vec3_t = SpectralPower::vec3_t;
	
	std::vector<vec3_t> isoVec;
	{
		if(n_request > 2)
		{
			// Ensuring that isoVec sums to zero is not trivial.
			// Adding up isotropic 3-vectors creates a 3D random walk, 
			// so while we expect (sum/n) to converge to zero, (sum) itself will diverge.
			// I have found 2 methods which keep sum balanced:
			// 	1. Quick and dirty; draw n/2 vectors, then add every vector's opposite.
			//		2. Monitor sum and whenever sum.Mag() gets too large,
			//			"shim" it by adding a unit vector opposite of sum. 
			//       Leave space for 2 unit vectors to neutralize the final sum.
			// Both of these methods have high-order (non-isotropic) correlations 
			// between vectors (as one would find in a multibody decay that
			// conserves momentum), but the second is clearly more isotropic.
			
			// We will keep a running sum of the vectors and 
			// verify the balance via sum.Mag2()
			double sum_Mag2;
			
			// How large should sum.Mag() be allowed to grow?
			// The balance vectors will be able to fix any sum.Mag() < 2.
			// If we requie sum.Mag() < 1., then we'll shim twice as often.
			double const sum_Mag2_max = kdp::Squared(2.)*(1.- tolerance);
			
			do // Keep regenerating isoVec until it is balanced (normally first times a charm)
			{				
				isoVec.clear();
				
				{
					// Draw the first vector and initialize the sum
					isoVec.push_back(IsoVec3(gen));
					vec3_t sum = isoVec.back();
					
					// Leave space for the two balancing vectors
					while(isoVec.size() < (n_request - 2))
					{
						isoVec.push_back(IsoVec3(gen));
						sum += isoVec.back();
						sum_Mag2 = sum.Mag2();
						
						// Do we need to "shim"?
						if(sum_Mag2 > sum_Mag2_max)
						{
							isoVec.push_back(-sum/std::sqrt(sum_Mag2));
							sum += isoVec.back();
						}
					}
					sum_Mag2 = sum.Mag2(); // Update (in case we exited after a shim)
							
					{
						// To create the pair of balancing vectors, 
						// we randomly find a vector orthogonal to sum.
						// The two balancing vectors will be 
						// 	a/b = -sum/2 +- ortho
						vec3_t ortho(false);
						
						// We checked externally that the following scheme 
						// produces a random orthogonal direction.
						{
							double ortho_Mag2;
						
							do // Keep finding ortho until its sufficiently orthogonal
							{
								// Cross with a random vector to find a random ortho
								ortho = sum.Cross(IsoVec3(gen));
								ortho_Mag2 = ortho.Mag2();
							}
							while(std::fabs(ortho.Dot(sum)) > 
								(tolerance * std::sqrt(ortho_Mag2 * sum_Mag2)));
							
							// To get |a| = |b| = 1, ortho_Mag2 = (1 - 0.25|sum|**2)
							// (from the defintion of a/b above). Scale ortho.
							ortho *= std::sqrt((1. - 0.25 * sum_Mag2)/ortho_Mag2);
						}
						
						// Add the two balancing vectors
						sum *= -0.5; // Done with sum, modify
						isoVec.push_back(sum + ortho);
						isoVec.push_back(sum - ortho);
					}
				}
						
				// Make sure the balancing vectors are really unit vectors
				// (assuming close to unit vector, so (1-tol)**2 ~= 1 - 2*tol).
				assert(std::fabs(isoVec.back().Mag2() - 1.) < 2. * tolerance);
				
				// Verify the consistency of sum by re-doing it ... backwards
				sum_Mag2 = 
					std::accumulate(isoVec.rbegin() + 1, isoVec.rend(), isoVec.back()).Mag2();
					
				// The error in the vector sum should grow like it's own random walk
				// (i.e. like the sqrt(n)), so we multiply the tolerance by sqrt(n).
				// This was noticed by counting how many times the magnitude was too high,
				// forcing regeneration. Now regeneration never happens.
			}
			while(sum_Mag2 > (tolerance * tolerance) * double(n_request));
		}
		else
		{
			switch(n_request)
			{
				case 2:
				{
					// We are simply choosing a random axis
					vec3_t const axis = IsoVec3(gen);
			
					isoVec.push_back(axis);
					isoVec.push_back(-axis);
				}
				break;
				
				case 1:
					isoVec.push_back(vec3_t()); // only the null vector fits the bill
				break;
				
				default: // n_request == 0, return an empty vector
				break;
			}
		}
	}
		
	return isoVec;
}

*/
