#include "NjetModel.hpp"
#include "RecursiveLegendre.hpp"
#include "pqRand/pqRand.hpp"
#include <algorithm> // std::max
#include <future>
#include "kdp/kdpTools.hpp"

////////////////////////////////////////////////////////////////////////

Jet::Jet(real_t const x1, real_t const x2, real_t const x3, 
	real_t const w0, kdp::Vec4from2 const w0type):
Jet(vec3_t(x1, x2, x3), w0, w0type) {}

////////////////////////////////////////////////////////////////////////

Jet::Jet(fastjet::PseudoJet const& pj):
	Jet(pj.px(), pj.py(), pj.pz(), pj.E(), kdp::Vec4from2::Energy) {}

////////////////////////////////////////////////////////////////////////

Jet::Jet(vec3_t const& p3_in, real_t const w0, kdp::Vec4from2 const w0type):
p4(w0, p3_in, w0type) // This will catch invalid w0 and throw exceptions
{
	switch(w0type)
	{
		case kdp::Vec4from2::Energy:
		case kdp::Vec4from2::Time:
			mass = p4.Length();
		break;
		
		case kdp::Vec4from2::Length:
		case kdp::Vec4from2::Mass:
			mass = w0;
		break;
		
		//~ case kdp::Vec4from2::Boost_preserve_p3:
		//~ case kdp::Vec4from2::Boost_preserve_E:
			//~ mass = p4.x0 / w0;
		//~ break;
		
		//~ case kdp::Vec4from2::BoostMinusOne_preserve_p3:
		//~ case kdp::Vec4from2::BoostMinusOne_preserve_E:
			//~ mass = p4.x0 / (w0 + real_t(1));
		//~ break;
	}
}

////////////////////////////////////////////////////////////////////////

void ShapedJet::SampleShape(incrementArray_t& z_lab, incrementArray_t& y_lab, 
	pqRand::engine& gen) const
{
	// We will boost z_CM into the lab frame
	// We assume boost collimates particles towards +z axis in the lab
	real_t const gamma2 = kdp::Squared(p4.x0 / mass);
	real_t const beta = vec4_t::Beta(p4.p(), mass);
	
	static constexpr size_t subIncrement = (incrementSize / 2);
	
	// This loop cannot be vectorized. There are a few reasons.
	// 1. U_uneven and ApplyRandomSign both have random branches visible to the compiler.
	// 2. sqrt and sin are functions whose control flow "cannot be analyzed".
	// 	sqrt is probably in hardware but sin is probably not, 
	//		and probably contains branches. I discovered #2 by separating 
	//		the PRNG to a warmup loop. The sqrt/sin loop still wasn't vectorized.
	// NOTE: in general, math function calls prohibit vectorization.
	for(size_t i = 0; i < subIncrement; ++i)
	{
		// Draw one u to generate antithetic z values (hopefully to reduce variance)
		real_t const u = gen.U_uneven(); 
		// But draw independent phi to avoid correlation between z and phi
		
		// z = z_lab =  (beta + z_CM)/(1 + beta * z_CM)	
		{
			// w+ = 1 - z = (1 - beta)(1 - z_CM)/(1 + beta * z_CM)
			real_t const w_plus = u / ((real_t(1) + beta) * gamma2 * 
				(real_t(1) + beta * (real_t(1) - u)));
			assert(w_plus > real_t(0));
			assert(w_plus <= real_t(2));
			
			z_lab[i] = real_t(1) - w_plus;
			y_lab[i] = std::sqrt(w_plus * (real_t(2) - w_plus)) * 
				gen.ApplyRandomSign(std::sin(gen.U_uneven() * M_PI_2));
		}
		
		{
			// w- = 1 + z 
			// this is the less accurate one
			real_t w_minus = u*(real_t(1) + beta)/(real_t(1) + beta*(u - real_t(1)));
			assert(w_minus > real_t(0));
			
			if(w_minus > real_t(2))
			{
				assert((w_minus - real_t(2)) < 1e-8);
				w_minus = real_t(2);
			}
			
			z_lab[subIncrement + i] = w_minus - real_t(1);
			y_lab[subIncrement + i] = std::sqrt(w_minus * (real_t(2) - w_minus)) * 
				gen.ApplyRandomSign(std::sin(gen.U_uneven() * M_PI_2));
		}
	}
}

bool ShapedJet::operator<(ShapedJet const& that) const
{
	//~ return (this->p4.x0 > that.p4.x0);
	//~ return (this->p4.x0 / this->mass) < (that.p4.x0 / that.mass);
	return this->mass > that.mass;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

ShowerParticle::ShowerParticle(ShowerParticle* mother_in, 
	vec3_t const& p3_in, real_t const mass_in, vec3_t const& pol_in, 
	std::vector<bool>&& address_in):
ShapedJet(p3_in, mass_in, kdp::Vec4from2::Mass, std::move(address_in)),
mother(mother_in),
b(nullptr), c(nullptr), 
pol(pol_in), inexact(false)
{}


std::vector<bool> ShowerParticle::DaughterAddress(bool const which) const
{
	std::vector<bool> copy(address);
	copy.push_back(which);
	return copy;
}

////////////////////////////////////////////////////////////////////////

void ShowerParticle::Split(param_iter_t const param_begin, param_iter_t param_end)
{
	splittingParams = std::vector<real_t>(param_begin, param_end);
		
	// We need at least 3 splitting parameters.  Get them last to first, 
	// because an exception will be thrown by at(2) if there are not enough.
	real_t const zStar = splittingParams.at(2);
	real_t const ubFrac = splittingParams[1]; // We now know that size is at least 3
	real_t const uSum = splittingParams[0];
	
	if((zStar > real_t(1)) or (zStar < real_t(0)))
		throw std::domain_error("ShowerParticle::Split: zStar must be in the inclusive unit interval.");
	if((ubFrac > real_t(1)) or (ubFrac < real_t(0)))
		throw std::domain_error("ShowerParticle::Split: u_b* must be in the inclusive unit interval.");
	if((uSum > real_t(1)) or (uSum < real_t(0)))
		throw std::domain_error("ShowerParticle::Split: u_sum must be in the inclusive unit interval.");		
	
	real_t const uDiff = (real_t(2) * ubFrac - real_t(1)) * uSum;
					
	vec3_t const& p3_a = p4.p();
	real_t const pSquared = p3_a.Mag2();
	real_t const massSquared = kdp::Squared(mass);
	
	// We find p3_b and the new polarization vector
	vec3_t p3_b(p3_a);
	vec3_t newPol(false);
	
				GCC_IGNORE_PUSH(-Wfloat-equal)			
	// Find how much of p3_b is parallel to p3_a
	{
		// If pSquared == 0, then b = inf, and b * p() == nan 
		// (even though we expect b * p() to be zero).
		real_t const r = (pSquared == real_t(0)) ? real_t(0) : 
			real_t(0.5)*(real_t(1) + uDiff * uSum + 
			(real_t(2) * zStar - real_t(1)) * 
			std::sqrt((massSquared + pSquared)/ pSquared * Delta2(uSum, uDiff)));
		assert(Delta2(uSum, uDiff) >= real_t(0));
		assert(not std::isnan(r));		
		
		p3_b *= r;
	}
				GCC_IGNORE_POP
	
	// Inside this scope there are three calls to Normalize()
	// The last two are crucial, the first one perhaps not so much.
	// However, since we can only remove 1/3 of them, leave them all.
	{
		// Get a's normalized direction of travel
		vec3_t const p_hat = p3_a / std::sqrt(pSquared);
		
		// If no phi is supplied, we assume phi = 0; pol does not rotate.
		if(splittingParams.size() == 3) // We've already ensured size >= 3
			newPol = pol;
		else
		{
			real_t const phi = splittingParams[3];
			
			// The polarization vector rotates by phi degrees.
			// 	rotated = cos * original + sin * transverse
			// To keep polarization consistent with the diagram in the thesis, 
			// we obtain the transverse unit vector via (pHat x pol) [opposite the original code]
			//~ vec3_t const kT_hat = pol.Cross(p_hat).Normalize();
			vec3_t const kT_hat = p_hat.Cross(pol).Normalize();
			// DO NOT silently enforce |phi| < pi/2 by forcing cos to be positive [as in the original code],
			// let the fitting routine correct large angles after the fact
			newPol = (pol * std::cos(phi) + kT_hat * std::sin(phi)).Normalize();
		}
		
		real_t const kT_mag = std::sqrt(zStar * (real_t(1) - zStar) * 
			kdp::Diff2(real_t(1), uSum) * kdp::Diff2(real_t(1), uDiff) * massSquared);
										
		p3_b += p_hat.Cross(newPol).Normalize() * kT_mag;
	}
	
	MakeDaughters(p3_b, ubFrac * uSum, uSum, newPol);
}
		
////////////////////////////////////////////////////////////////////////

void ShowerParticle::MakeDaughters(vec3_t const& p3_b, 
	real_t const u_b, real_t const uSum, vec3_t const& newPol)
{
	b = new ShowerParticle(this, p3_b, u_b * mass, newPol, DaughterAddress(false));
	
	// Got a negative mass errors (machine epsilon subtraction error)
	// when ubFrac = 1. This code will prevent future occurrences.
	real_t u_c = uSum * mass - b->mass;
	assert(u_c > -real_t(1e-15)*mass);
	if(u_c < real_t(0))
		u_c = real_t(0);
	
	c = new ShowerParticle(this, p4.p() - p3_b, u_c, newPol, DaughterAddress(true));
	
	//~ b = std::shared_ptr<ShowerParticle>(new ShowerParticle(this, p3_b, u_b * mass, newPol));
	//~ c = std::shared_ptr<ShowerParticle>(new ShowerParticle(this, p4.p() - p3_b, uSum * mass - b->mass, newPol));
	
					GCC_IGNORE_PUSH(-Wfloat-equal)			
	/* With floating point arithmetic, we can guarantee that
	 * 	(larger - smaller) + smaller = larger
	 * So because we defined p_c by subtraction, we can guarantee momentum conservation.
	 * What we cannot guarantee is that energy is conserved, 
	 * since we define the two daughters via their 3-momentum and mass, 
	 * and these *should* add up to the original mass, but there is no guarantee
	*/
	inexact = (EnergyLoss_unsafe() not_eq real_t(0));
					GCC_IGNORE_POP
			
	//~ printf("%.16e, %.16e\n", sum.x0, p4.x0);
	//~ printf("%.16e\n", kdp::RelDiff(sum.x0, p4.x0));
	//~ assert(kdp::AbsRelDiff(sum.x0, p4.x0) < 10.*std::numeric_limits<real_t>::epsilon());
	// All 4-momentum worked out as it should
	//~ printf("(%.16e, %.16e, %.16e, %.16e)\n", sum.x0, sum.x1, sum.x2, sum.x3);
	//~ assert((p4 - sum).Length() < 10.*std::numeric_limits<real_t>::epsilon());
}

////////////////////////////////////////////////////////////////////////

void ShowerParticle::AppendJets(std::vector<ShapedJet>& existing)
{
	if(isLeaf())
		existing.push_back(*this);
	else
	{
		b->AppendJets(existing);
		c->AppendJets(existing);
	}		
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::real_t ShowerParticle::Delta2(real_t const uSum, real_t const uDiff)
{
	return kdp::Diff2(real_t(1), uSum)*kdp::Diff2(real_t(1), uDiff);
}

////////////////////////////////////////////////////////////////////////

std::string ShowerParticle::AddressToString(std::vector<bool> const& address)
{
	std::string addStr;
	
	for(bool const node : address)
		addStr += (node ? "1" : "0");
		
	return addStr;
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::address_error ShowerParticle::NoSuchAddress
	(std::vector<bool> const& address, size_t const level)
{
	return address_error("ShowerParticle::address_error. No such address <" 
		+ AddressToString(address) + ">; fails at level " + std::to_string(level) + ".");
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::address_error ShowerParticle::AddressAlreadySplit
	(std::vector<bool> const& address)
{
	return address_error("ShowerParticle::address_error. Address <" 
		+ AddressToString(address) + "> already split, cannot split again.");
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::ShowerParticle(std::vector<real_t> const& params, 
	std::vector<std::vector<bool>> const& addresses, bool const orientation):
// Build the root node in its CM frame with energy = 1
ShapedJet(vec3_t(), real_t(1), kdp::Vec4from2::Mass),
mother(nullptr), b(nullptr), c(nullptr),
pol(), inexact(false)
{
	/* There are two formats for params
	 * 
	 * no orientation (2, 3, then 4 per splitting):
	 * 
	 * 	params = {
	 * 		u_sum, u_b*, 
	 * 		u_sum, u_b*, z*,
	 *       u_sum, u_b*, z*, phi,
	 * 		etc}
	 * 
	 * orientation (4 params per splitting)
	 * 	
	 * 	params = {
	 * 		u_sum, u_b*, theta', phi', 
	 * 		u_sum, u_b*, z*, omega',
	 *       u_sum, u_b*, z*, phi,
	 * 		etc}
	 * 
	 * The first parameters never has an address because they must be the root.
	 * The next bit of code sorts between these two options and acts accordingly. 
	*/ 
	
	using vec3_t = ShowerParticle::vec3_t;
	
	// If we don't care about orientation, then the choice of
	// original splitting axis and orientation is arbitrary (they must be perpendicular).
	vec3_t dijetAxis(0., 0., 1.);
	vec3_t dijetPol(1., 0., 0.);
	
	static constexpr size_t numRootParams = 2; // Num params to specify the root branching
	static constexpr size_t numNextParams = 3; // Num params to specify the next branching
	
	if(orientation)
	{
		// 4 parameters for every splitting, including the root splitting
		if((4 * (addresses.size() + 1)) not_eq params.size())
			throw std::runtime_error("ShowerParticle: Number of parameters supplied does not match number of addresses" +
				std::string("(with orientation, 4 for every particle)."));
		
		// omega defines the orientation of the first splitting plane
		// either it's the 8th parameter or it's zero (because we don't split twice)
		real_t const omega = bool(addresses.size()) ? params[7] : real_t(0);
		
		// We rotate z^ to some off-axis position, then rotate about this axis by omega
		kdp::Rotate3<real_t> rot(dijetAxis, vec3_t(1., params[2], params[3], 
			kdp::Vec3from::LengthThetaPhi), omega);
		
		rot(dijetAxis);
		rot(dijetPol);
	}
	else
	{
		if((addresses.empty() and (params.size() not_eq numRootParams)) or 
			(addresses.size() and (4 * addresses.size() - 1) not_eq (params.size() - numRootParams)))
		{
			throw std::runtime_error("ShowerParticle: Number of parameters supplied does not match number of addresses" +
				std::string("(2 for the first particle, 3 for the second, 4 for all subsequent)."));
		}
	}
	
	using param_it = std::vector<real_t>::const_iterator;
		
	// For all branches after the root branching, store the first and last parameter iterator
	std::vector<std::pair<param_it, param_it>> param_begin_end;
	
	if(addresses.size())
	{
		if(orientation)
			param_begin_end.emplace_back(params.begin() + 4, 
				params.begin() + 4 + numNextParams);
		else
			param_begin_end.emplace_back(params.begin() + numRootParams, 
				params.begin() + numRootParams + numNextParams);
		
		for(size_t i = 1; i < addresses.size(); ++i)
		{
			auto first = param_begin_end.back().second;
			if ((i == 1) and orientation) 
				++first;
				
			param_begin_end.emplace_back(first, first + 4);
		}
				
		// Verify that we've used all the parameters
		assert(param_begin_end.size() == addresses.size());
		
		if(addresses.size() == 1)
			assert(param_begin_end.back().second == (params.end() - (orientation ? 1 : 0)));
		else
			assert(param_begin_end.back().second == params.end());
	}
	
	// The root splitting needs 2 parameters
	if(params.size() >= numRootParams)
	{
		//~ splittingParams = std::vector<real_t>(params.cbegin(), 
			//~ params.cbegin() + numRootParams);
			
		/* We treat the root splitting differently, to mantain the correct toplogy
		 * (we want the top two partons on either side of the event).
		 * 	f_{b/c} = 1/2 (1 +/- (u_b-u_c)(u_b+u_c))
		 * and we want f_{b/c} > 1/3. Thus we want
		 * 	u2 = (u_b - u_c)*(u_b + u_c)
		 * 	|u2| < 1 / 3
		 * Using u_sum = u_b + uc, we can solve for
		 * 	u_{b/c} = 1/2(u_sum +/- u2 / u_sum)
		 * which sets a limit on u_sum
		 * 	u_sum^2 > u2
		 * And this allows us to define
		 * 	u_sum = sqrt(fabs(u2)) + u*_sum * (1 - sqrt(fabs(u2)))
		 * 	0 <= u*_sum <= 1
		*/ 
		//~ real_t const u2 = splittingParams[0];
		//~ real_t const uSumStar = splittingParams[1];
		
		//~ real_t const uSum_min = std::sqrt(std::fabs(u2));
		//~ real_t const uSum = uSum_min + (real_t(1) - uSum_min)*uSumStar;
		
		//~ real_t const uDiff = (uSum == real_t(0)) ? real_t(0) : u2 / uSum;
		
		//~ MakeDaughters(vec3_t(0., 0., 0.5*std::sqrt(Delta2(uSum, uDiff))),
			//~ real_t(0.5)*(uSum + uDiff), uSum, vec3_t(1., 0., 0.));
		
		real_t const uSum = params[0];
		real_t const ubFrac = params[1];
		real_t const uDiff = (real_t(2)*ubFrac - real_t(1))*uSum;
		
		//~ printf("%.16e, %.16e\n", uSum, uDiff);
		
		// The CM mommentum is 0.5*std::sqrt(Delta2(uSum, uDiff)) 
		MakeDaughters(dijetAxis * 0.5 * std::sqrt(Delta2(uSum, uDiff)),
			ubFrac * uSum, uSum, dijetPol);
			
		for(size_t i = 0; i < addresses.size(); ++i)
		{
			ShowerParticle& toSplit = LocateParticle(addresses[i]);
				
			if(toSplit.isBranch())
				throw AddressAlreadySplit(addresses[i]);
			
			toSplit.Split(param_begin_end[i].first, param_begin_end[i].second);
		}
	}
}

////////////////////////////////////////////////////////////////////////

ShowerParticle& ShowerParticle::operator=(ShowerParticle&& orig)
{
	std::swap(mother, orig.mother);
	std::swap(b, orig.b);
	std::swap(c, orig.c);
	return *this;
}		

////////////////////////////////////////////////////////////////////////

ShowerParticle::~ShowerParticle()
{
	delete b;
	delete c;
}

////////////////////////////////////////////////////////////////////////

bool ShowerParticle::isBranch()
{
	// Consistency check; either both b and c are initialized, or they are both nullptr
	assert(bool(b) == bool(c));
	return bool(b);
}

////////////////////////////////////////////////////////////////////////

bool ShowerParticle::isShowerInexact()
{
	if(isLeaf()) // A leaf cannot be inexact
		return false;
	else
	{
		if(inexact) // Stop the search once we find the first inexact splitting
			return true;
		else
			return (b->isShowerInexact() or c->isShowerInexact());
	}
}

////////////////////////////////////////////////////////////////////////

ShowerParticle& ShowerParticle::LocateParticle(std::vector<bool> const& theAddress)
{
	// Starting here, navigate down until we locate address
	ShowerParticle* currentNode = this;
	// The level is the index of the current bool
	size_t level = 0; // Slower than an iterator, but more readible. This function is not what takes the time.
	
	while(level < theAddress.size())
	{
		// If currentNode is a leaf, we can't derefence b or c; this address makes no sense.
		if(currentNode->isLeaf())
			throw NoSuchAddress(theAddress, level);
		else
			currentNode = theAddress[level++] ? currentNode->c : currentNode->b;
			//~ currentNode = address[level++] ? currentNode->c.get() : currentNode->b.get();
			// Since we are not going to delete currentNode, it is same to use pointers directly
	}
	assert(currentNode->address == theAddress);
	
	return *currentNode;
}

////////////////////////////////////////////////////////////////////////

std::vector<ShapedJet> ShowerParticle::GetJets()
{
	std::vector<ShapedJet> retVec;			
	AppendJets(retVec);
	
	return retVec;
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::real_t ShowerParticle::EnergyLoss()
{
	if(isLeaf())
		return real_t(0);
	else
		return EnergyLoss_unsafe();
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::vec4_t ShowerParticle::Total_p4()
{
	std::vector<vec4_t> p4_vec;
	
	for(auto const& jet : GetJets()) // Convert jets to 4-vectors
		p4_vec.push_back(jet.p4);
		
	return kdp::BinaryAccumulate_Destructive(p4_vec);
}

////////////////////////////////////////////////////////////////////////

ShowerParticle::real_t ShowerParticle::Total_absElost(real_t const absElost_in)
{
	if(isLeaf())
		return real_t(0);
	else
		return absElost_in + (std::fabs(EnergyLoss()) + 
			(b->Total_absElost() + c->Total_absElost()));
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

//~ NjetModel::NjetModel(QSettings const& settings):
	//~ gen(false)
//~ {
	//~ if(settings.contains("NjetModel/seedPath"))
	//~ {
		//~ const std::string seedPath = settings.value("NjetModel/seedPath").toString().toStdString();
		
		//~ if(settings.value("NjetModel/writeSeed", true).toBool())
		//~ {
			//~ gen.Seed();
			//~ gen.WriteState(seedPath);
		//~ }
		//~ else
			//~ gen.Seed_FromFile(seedPath);
	//~ }
	//~ else
		//~ gen.Seed();
//~ }

////////////////////////////////////////////////////////////////////////

//~ NjetModel::NjetModel(std::string const& iniFileName):
	//~ NjetModel(QSettings(iniFileName.c_str(), QSettings::IniFormat)) {}

//~ ////////////////////////////////////////////////////////////////////////

//~ NjetModel::NjetModel():NjetModel(QSettings()) {}

//~ ////////////////////////////////////////////////////////////////////////

//~ NjetModel::~NjetModel() {}//delete detector;}

////////////////////////////////////////////////////////////////////////

//~ std::vector<std::vector<NjetModel::real_t>> NjetModel::DoIncrements_jet_i
	//~ (size_t const i, size_t const lMax, std::vector<ShapedJet> const& jetVec,
	//~ //kdp::MutexCount<size_t>& kShared, 
	//~ size_t const numIncrements,
	//~ std::string const& generator_seed, 
	//~ bool const onlySelf)
//~ {
	//~ pqRand::engine gen(false); // Don't auto-seed; gen is unitialized
	//~ gen.Seed_FromString(generator_seed);
	
	//~ using incrementArray_t = std::array<real_t, ShapedJet::incrementSize>;
	
	//~ RecursiveLegendre_Increment<incrementArray_t> Pl_computer;	
	
	//~ // Each jet will fill shape variate positions into these two arrays
	//~ incrementArray_t z_lab, y_lab;
	
	//~ size_t const j_begin = (onlySelf ? i : 0);

	//~ // To avoid the redundant calculation of inter-jet cos/sin as we 
	//~ // repeatedly loop over the same j, we calculate then during
	//~ // the first increment and reuse it for all others.
	//~ std::vector<std::pair<real_t, real_t>> cos_sin_ij;
	//~ bool firstIncrement = true;
		
	//~ std::vector<std::vector<real_t>> rho_accumulate((i - j_begin) + 1, 
		//~ std::vector<real_t>(lMax, 0));
		
	//~ ShapedJet const& jet_i = jetVec[i];	
	
	//~ for(size_t k = 0; k < numIncrements; ++k) // loop over increments
	//~ {
		//~ // These are the positions when jet_i is boosted to the lab, 
		//~ // but still parallel to the z axis. They must be rotated off-axis
		//~ // for each jet_j.		
		//~ jet_i.SampleShape(z_lab, y_lab, gen);
		
		//~ for(size_t j = j_begin; j <= i; ++j)
		//~ {
			//~ if(firstIncrement) // Find and cache interior angle
			//~ {
				//~ if(j == i)
					//~ cos_sin_ij.emplace_back(1, 0);
				//~ else
					//~ cos_sin_ij.push_back(CosSin(jet_i.p4.p(), jetVec[j].p4.p()));
			//~ }
			//~ auto const& cos_sin = cos_sin_ij[j];
			
			//~ // WARNING. May want to check that |z| <= 1, otherwise recursive singularity
			//~ for(size_t m = 0; m < ShapedJet::incrementSize; ++m)
				//~ Pl_computer.z[m] = cos_sin.first * z_lab[m] + cos_sin.second * y_lab[m];
				
			//~ Pl_computer.Reset();
			
			//~ // loop over l, starting with l=1
			//~ for(auto& rho_accumulate_j_l : rho_accumulate[j - j_begin])
			//~ {
				//~ Pl_computer.Next(); // To prevent unnecessary calculation, run at start of loop
				//~ rho_accumulate_j_l += kdp::BinaryAccumulate(Pl_computer.P_lm1()); // l is one ahead, use lm1
				//~ assert(not std::isnan(rho_accumulate_j_l));
			//~ }
		//~ }
		//~ firstIncrement = false; // After looping over all j, we have cached all cos_sin
	//~ }
	
	//~ return rho_accumulate;
//~ }

////////////////////////////////////////////////////////////////////////

//~ std::pair<NjetModel::real_t, NjetModel::real_t> NjetModel::CosSin(vec3_t const& a, vec3_t const& b)
//~ {
	//~ real_t const mag2 = a.Mag2() * b.Mag2();

	//~ std::pair<real_t, real_t> cos_sin(
		//~ a.Dot(b)/std::sqrt(mag2), 
		//~ std::sqrt(a.Cross(b).Mag2()/mag2));
	
	//~ // Check for over-unity from rounding (but they can't BOTH be over-unity)
	//~ // Rare, so who cares that round is slower than copysign
	//~ if(std::abs(cos_sin.first) > real_t(1))
		//~ cos_sin.first = std::round(cos_sin.first);
	//~ else if((std::abs(cos_sin.second) > real_t(1)))
		//~ cos_sin.second = std::round(cos_sin.second);
		
	//~ return cos_sin;
//~ }

////////////////////////////////////////////////////////////////////////

//~ std::vector<std::vector<NjetModel::real_t>> NjetModel::rho_j_l(
	//~ size_t const i, size_t const lMax,
	//~ real_t const jetShapeGranularity, real_t const Etot,
	//~ std::vector<ShapedJet> const& jetVec_sorted,
	//~ bool const onlySelf) const
//~ {
	//~ std::vector<std::vector<real_t>> rho; // The rho for each jet_j (and each l)
	
	//~ // The number of variates can be very large, so we obtain
	//~ // them in increments, and for each increment loop over jet_j.
	//~ // This minimizes the memory overhead.
	//~ // Additionally, since each increment/j-loop can be done separately, 
	//~ // we can launch a number of threads which accomplish many increments each.
	//~ static constexpr size_t numThreads_max = 4; // hard-coded for the time being
	//~ static constexpr size_t const minIncrements_perThread = 1; // 1 is fine, given a large enough increment size
	
	//~ // Originally, threads shared a count, incrementing until kShared++ == n_increments.
	//~ // However, this leads to a non-deterministic result
	//~ // (given the same seed of NjetModel.gen)
	//~ // because we don't know how many increments each thread will 
	//~ // actually handle at runtime, and each thread uses its own PRNG. 
	//~ // If we require each thread to handle the same number of increments,
	//~ // then it will always call the PRNG the same number of times.
	//~ // kdp::MutexCount<size_t> kShared(0);
				
	//~ ShapedJet const& jet_i = jetVec_sorted[i];
	
	//~ size_t const n_requested = 
		//~ std::max(1lu, // Must have a sample size of at least 1
			//~ // (jet_i->p4.x0 < 1e3*jet_i->mass) ? 0 : 
				//~ size_t(jet_i.p4.x0 * jetShapeGranularity / Etot));
	//~ size_t const n_increments = kdp::MinPartitions(n_requested, ShapedJet::incrementSize);
	//~ size_t const numThreads = std::max(1lu, std::min(numThreads_max, 
		//~ n_increments / minIncrements_perThread));
	//~ size_t const n_increments_per_thread = kdp::MinPartitions(n_increments, numThreads);
	
	//~ size_t const sampleSize = ShapedJet::incrementSize * 
		//~ n_increments_per_thread * numThreads;
	
	//~ {
		//~ // Each thread accumulates a matrix rho_j_l for jet_i.
		//~ // We keep each jet_j separate because they each have there own normalization,
		//~ // and we would prefer to normalize as little as possible (less FLOPS).
		//~ // std::vector<std::vector<real_t>> rho_accumulate;
		
		//~ // We need to use std::future for each thread's return value.
		//~ // Each thread will return a 2D matrix of floats.
		//~ std::vector<std::future<
			//~ std::vector<std::vector<real_t>>>> threadReturn;
			
		//~ // Each thread should have it's own PRNG (for efficiency and determinism).
		//~ // Instead of trying to pass PRNG objects, pass the state string.
		//~ // We will jump the classes's main generator each time we spawn a new thread.
		//~ // This is slightly wasteful (we don't need a full Jump() of 2**512 calls), 
		//~ // but it's the simplest method that is totally deterministic.
		//~ // Since we can Jump() the generator 2**512 times, we can't over-jump.
		//~ std::vector<std::string> seedVec = gen.GetState_JumpVec(numThreads);
		
		//~ // Create/launch all threads and bind their return value
		//~ for(size_t t = 0; t < numThreads; ++t)
		//~ {
			//~ // A few notes here:
			//~ // 1. member pointer must be &class::func not &(class::func)
			//~ // 	https://stackoverflow.com/questions/7134197/error-with-address-of-parenthesized-member-function
			//~ // 2. Any object which is to be passed by reference must use std::ref
			//~ threadReturn.push_back(
				//~ std::async(std::launch::async, &NjetModel::DoIncrements_jet_i,
				//~ i, lMax, 
				//~ std::cref(jetVec_sorted), 
				//~ // std::ref(kShared), 
				//~ n_increments_per_thread, 
				//~ std::cref(seedVec[t]), onlySelf)); // true = only self rho, but we want all
		//~ }
		
		//~ for(size_t t = 0; t < numThreads; ++t)
		//~ {
			//~ // Get the result (get() will block until this tread's result is ready, 
			//~ // and get() can only be called once).
			//~ std::vector<std::vector<real_t>> rho_thread = threadReturn[t].get();
			
			//~ // Initialize rho_accumulate to the first thread (steal its data).
			//~ if(t == 0) 
				//~ rho = std::move(rho_thread);
			//~ else
			//~ {
				//~ // For all other threads, loop over j, then l
				//~ for(size_t j = 0; j < rho.size(); ++j)
					//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
						//~ rho[j][lMinus1] += rho_thread[j][lMinus1];
			//~ }
		//~ }
	//~ }
	
	//~ // Do jet_i's energy and sample-size normalization once (to reduce FLOPS and rounding error)
	//~ {
		//~ // Moved Etot normalization here for more generic use of this function
		//~ // (i.e. so it actually returns what it claims to calculate).
		//~ real_t const normalization = jet_i.p4.x0 / (real_t(sampleSize) * Etot);
		
		//~ for(size_t j = 0; j < rho.size(); ++j)
		//~ {
			//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
			//~ {
				//~ rho[j][lMinus1] *= normalization;
				//~ assert(not std::isnan(rho[j][lMinus1]));
			//~ }
		//~ }
	//~ }
	
	//~ return rho;
//~ }

////////////////////////////////////////////////////////////////////////

//~ std::vector<ShapedJet> NjetModel::SortBy_E(std::vector<ShapedJet> jetVec_unsorted)
//~ {
	//~ // First sort the incoming jetVec from high to low energy.
	//~ std::vector<ShapedJet> jetVec_sorted = jetVec_unsorted;
	
	//~ // We can use lambdas (no-capture, empty square bracket) to use some standard tools with Jet class
	//~ std::sort(jetVec_sorted.begin(), jetVec_sorted.end(), 
		//~ [](ShapedJet const& lhs, ShapedJet const& rhs) {return lhs.p4.x0 > rhs.p4.x0;});
		
	//~ return jetVec_sorted;
//~ }

//~ NjetModel::real_t NjetModel::Total_E(std::vector<ShapedJet> jetVec)
//~ {
	//~ // Add up energy from back to front (small to large)
	//~ return std::accumulate(jetVec.crbegin(), jetVec.crend(), real_t(0), 
		//~ // a lambda which sums jet energy for a std::accumulate
		//~ [](real_t const E_current, ShapedJet const& jet) {return E_current + jet.p4.x0;});
//~ }

////////////////////////////////////////////////////////////////////////

//~ std::vector<NjetModel::real_t> NjetModel::H_l
	//~ (std::vector<ShapedJet> const& jetVec_unsorted, 
	//~ size_t const lMax, real_t const jetShapeGranularity) const
//~ {
	//~ // Return H_l from (l = 1) to (l = lMax)
	//~ std::vector<real_t> H_l_vec(lMax, 0.);
	
	//~ /* Defining the integral
	 //~ * 	rho_(i) = f_i * h(z) / (2 pi)
	 //~ * 	rho_(i)_l = \int P_l(z) rho_(i) dz dphi = f_i \int P_l(z) h(z) dz
	 //~ * We need to calculate
	 //~ * 	H_l = Etot**(-2)*((rho_(1)_l)**2 + rho_(1)_l * rho_(2)_l + ... + rho_(2)_l * rho_(1)_l + ... )
	 //~ * However the integral is difficult. Instead, we use 
	 //~ * Monte Carlo integration for n variates drawn from h(z)
	 //~ * 	\rho_(i)_l ~= 1/n * sum_k P_l(z_k) * h(z_k) / h(z_k) = 1/n * sum_k P_l(z_k)
	 //~ * We can compute each cross-term {rho_(i)_l * rho_(j)_l} as 
	 //~ * {rho_(i,j)_l * rho_(j)_l}
	 //~ * 	a) compute rho_(j)_l with jet_j parallel to the z-axis,
	 //~ * 		and merely boosted from its CM frame to the lab frame.
	 //~ *		b) compute rho_(i,j)_l with jet_i starting parallel to the z-axis, 
	 //~ * 		boosted into the lab frame along the z-axis, 
	 //~ * 		then rotated off axis by the ij interior angle.
	 //~ * This naively requires N = (n_jets * n)**2 / 2 variates and Order(N * lMax) FLOPS.
	 //~ * However, this is quite redundant, as we can reuse variates.
	 //~ * 
	 //~ * 1. For each jet_i, draw n variates.
	 //~ * 2. For j < i, jet_j is the jet parallel to the z-axis, 
	 //~ * 	and jet_i is the rotated jet. Compute rho_(i,j)_l and 
	 //~ * 	multiply by the pre-computed rho_(j)_l (which only needs to be 
	 //~ * 	computed once, since it does not depend on inter-jet angles).
	 //~ * 3. When j == i, compute rho_(j)_l for use by all subsequent i.
	 //~ * 
	 //~ * Since n is proportional to energy (because smaller jets are less important),
	 //~ * sorting jets from largest energy to smallest minimizes the total 
	 //~ * number of FLOPS (because we need Order(n * lMax) FLOPS per cross-term, 
	 //~ * and there is one additional cross-term each time i increments).
	//~ */ 
	//~ if(lMax > 0)
	//~ {
		//~ std::vector<ShapedJet> jetVec = SortBy_E(jetVec_unsorted);
		
		//~ real_t const Etot = Total_E(jetVec);
		//~ if(Etot <= real_t(0))
			//~ throw std::runtime_error("NjetModel::H_l: zero energy jets supplied; most likely you did not want this.");		
		
		//~ std::vector<std::vector<real_t>> rho_self; // The rho for each jet_j (and each l)
				
		//~ // In the following outer loops, we will use i and j because it makes
		//~ // the code more readable. Being outer loops, the penalty is tiny.
		//~ for(size_t i = 0; i < jetVec.size(); ++i)
		//~ {
			//~ {
				//~ std::vector<std::vector<real_t>> rho_i_j_l = rho_j_l(i, lMax, 
					//~ jetShapeGranularity, Etot, jetVec, false); // false = do j = 0 through i
				
				//~ for(size_t j = 0; j < i; ++j)
					//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
						//~ H_l_vec[lMinus1] += real_t(2) * // two symmetric cross-terms
							//~ rho_self[j][lMinus1] * rho_i_j_l[j][lMinus1];
				
				//~ // Steal the rho for jet_i (the self rho)
				//~ rho_self.emplace_back(std::move(rho_i_j_l.back()));
			//~ }
				
			//~ auto& rho_i = rho_self.back();
			
			//~ // Add the self contribution; 
			//~ // We can't both steal rho_i_j_l.back() and place this operation in the other loop
			//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
			//~ {
				//~ H_l_vec[lMinus1] += (kdp::Squared(rho_i[lMinus1]));
				//~ assert(not std::isnan(H_l_vec[lMinus1]));
			//~ }
		//~ }
	//~ }
	
	//~ return H_l_vec;
//~ }

//~ NjetModel::JetParticle_Cache::JetParticle_Cache(NjetModel const& modeler,
	//~ std::vector<ShapedJet> const& jetVec_in,
	//~ size_t const lMax_in, real_t const jetShapeGranularity):
//~ jetVec(SortBy_E(jetVec_in))
//~ {
	//~ real_t const Etot = Total_E(jetVec);
	
	//~ for(size_t i = 0; i < jetVec.size(); ++i)
	//~ {
		//~ rho_jet.push_back(
			//~ modeler.rho_j_l(i, lMax_in, 
				//~ jetShapeGranularity, Etot, jetVec, true) // true = only do self rho
			//~ .front()); // because "only do self rho", there is only one element in the returned vector
	//~ }


//~ }

//~ NjetModel::JetParticle_Cache NjetModel::Make_JetParticle_Cache
	//~ (std::vector<ShapedJet> const& jetVec,
	//~ size_t const lMax, real_t const jetShapeGranularity) const
//~ {
	//~ return NjetModel::JetParticle_Cache(*this, jetVec, lMax, jetShapeGranularity);
//~ }

//~ std::vector<typename NjetModel::real_t> NjetModel::H_l_JetParticle
//~ (JetParticle_Cache const& cache, std::vector<SpectralPower::PhatF> const& particles, 
	//~ vec3_t const& axis, real_t const angle) const
//~ {
	//~ std::vector<ShapedJet> jets_labFrame = cache.jetVec; // copy all jets, so we can modify
	//~ Jet::Rotate(jets_labFrame, axis, angle);
		
	//~ // Number of particles probably O(100)
	//~ static constexpr size_t incrementSize = (size_t(1) << 5);	// 32
	//~ using array_t = std::array<real_t, incrementSize>;
	//~ RecursiveLegendre_Increment<array_t> Pl_computer;
	
	//~ size_t const lMax = cache.lMax();
	//~ std::vector<real_t> H_l_vec(lMax, real_t(0));	
	
	//~ // Parallelize this only if it's slowing us down	
	//~ for(size_t i = 0; i < jets_labFrame.size(); ++i)
	//~ {
		//~ std::vector<real_t> rho_i_particle(lMax, real_t(0)); // rho_particles depends on the jet i. 
		//~ array_t rho_i_particles_increment; 
		//~ array_t f_particle;
		
		//~ vec3_t const jet_i_hat = vec3_t(jets_labFrame[i].p4.p()).Normalize();
		
		//~ for(size_t n = 0; n < particles.size(); n += incrementSize)
		//~ {
			//~ {
				//~ size_t k = 0;
				//~ for(; k < std::min(incrementSize, particles.size() - n); ++k)
				//~ {
					//~ Pl_computer.z[k] = jet_i_hat.Dot(particles[n + k].pHat);
					//~ f_particle[k] = particles[n + k].f;
				//~ }
					
				//~ // Zero fill the last increment
				//~ for(; k < incrementSize; ++k)
					//~ f_particle[k] = Pl_computer.z[k] = real_t(0);
			//~ }
					
			//~ Pl_computer.Reset();
				
			//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
			//~ {
				//~ Pl_computer.Next();
				
				//~ for(size_t k = 0; k < incrementSize; ++k)
					//~ rho_i_particles_increment[k] = f_particle[k] * Pl_computer.P_lm1()[k];
					
				//~ rho_i_particle[lMinus1] += kdp::BinaryAccumulate(rho_i_particles_increment);
			//~ }
		//~ }
		
		//~ for(size_t lMinus1 = 0; lMinus1 < lMax; ++lMinus1)
			//~ H_l_vec[lMinus1] += cache.rho_jet[i][lMinus1] * rho_i_particle[lMinus1];
	//~ }
	
	//~ return H_l_vec;
//~ }
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
			
			double const energy = std::sqrt(p3.Mag2() + Squared(mass));
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
			double const sum_Mag2_max = Squared(2.)*(1.- tolerance);
			
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
