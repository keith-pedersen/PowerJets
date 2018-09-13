// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PowerSpectrum.hpp"
#include "RecursiveLegendre.hpp"
#include "ShapeFunction.hpp"
#include "kdp/kdpStdVectorMath.hpp"
#include <future>

////////////////////////////////////////////////////////////////////////
// PhatF
////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(vec3_t const& p3):
	pHat(p3), f(p3.Mag())
{
	pHat /= f; // Normalize pHat
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
	pHat.Normalize(); // Don't use f to normalize, in case the particle has a small mass
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatF::PhatF(Pythia8::Particle const& particle):
	PhatF(particle.px(), particle.py(), particle.pz(), particle.e()) {}
	
////////////////////////////////////////////////////////////////////////
// ParticleContainer
////////////////////////////////////////////////////////////////////////

void PowerSpectrum::ParticleContainer::NormalizeF()
{
	NormalizeF_to(fTotal());
}

////////////////////////////////////////////////////////////////////////

bool PowerSpectrum::ParticleContainer::IsNormalized(real_t const threshold) const
{
	// Instead of ensuring that everything is exactly 1, 
	// ensure that it is close enough (within the supplied rounding-error threshold).
	// Return false the first time something isn't normalized	
	
	if(std::fabs(fTotal() - real_t(1)) >= threshold)
		return false;
		
	for(size_t i = 0; i < size(); ++i)
	{
		if(std::fabs(pHatMag(i) - real_t(1)) >= threshold)
			return false;
	}
	
	return true;
}

////////////////////////////////////////////////////////////////////////
// VecPhatF
////////////////////////////////////////////////////////////////////////
			
PowerSpectrum::real_t PowerSpectrum::VecPhatF::pHatMag(size_t const i) const
{
	// We use at so we can catch (via runtime error) index OOB
	return this->at(i).pHat.Mag();
}

////////////////////////////////////////////////////////////////////////
			
void PowerSpectrum::VecPhatF::NormalizeF_to(real_t const total_f)
{
	for(PhatF& particle : *this)
		particle.f /= total_f;
}

////////////////////////////////////////////////////////////////////////
			
PowerSpectrum::real_t PowerSpectrum::VecPhatF::fTotal() const
{
	return std::accumulate(this->cbegin(), this->cend(), 
		real_t(0), &Accumulate_f);
}

////////////////////////////////////////////////////////////////////////
			
PowerSpectrum::real_t PowerSpectrum::VecPhatF::fInner() const
{
	return std::accumulate(this->cbegin(), this->cend(), 
		real_t(0), &Accumulate_f2);
}

////////////////////////////////////////////////////////////////////////
			
PowerSpectrum::real_t PowerSpectrum::VecPhatF::Accumulate_f(real_t const sum,
	PhatF const& particle)
{
	return sum + particle.f;
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::VecPhatF::Accumulate_f2(real_t const sum, 
	PhatF const& particle)
{
	return sum + kdp::Squared(particle.f);
}

////////////////////////////////////////////////////////////////////////
// PhatFvec
////////////////////////////////////////////////////////////////////////

PowerSpectrum::PhatFvec::PhatFvec(std::vector<PhatF> const& orig)
{
	reserve(orig.size());
	for(PhatF const& p : orig)
		this->emplace_back(p);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::PhatFvec::pHatMag(size_t const i) const
{
	return std::sqrt(kdp::Squared(x.at(i)) + kdp::Squared(y.at(i)) + kdp::Squared(z.at(i)));
}

////////////////////////////////////////////////////////////////////////
			
void PowerSpectrum::PhatFvec::NormalizeF_to(real_t const total_f)
{
	for(real_t& f_i : f)
		f_i /= total_f;
}

////////////////////////////////////////////////////////////////////////
			
PowerSpectrum::real_t PowerSpectrum::PhatFvec::fTotal() const
{
	return std::accumulate(f.cbegin(), f.cend(), real_t(0));
}

////////////////////////////////////////////////////////////////////////
			
PowerSpectrum::real_t PowerSpectrum::PhatFvec::fInner() const
{
	return std::accumulate(f.cbegin(), f.cend(), real_t(0), 
	// Use a quick lambda expression instead of defining an extra function
		[](real_t const sum, real_t const val){return sum + kdp::Squared(val);});
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::PhatFvec::reserve(size_t const reserveSize)
{
	x.reserve(reserveSize);
	y.reserve(reserveSize);
	z.reserve(reserveSize);
	
	f.reserve(reserveSize);
}

//~ ////////////////////////////////////////////////////////////////////////

//~ void PowerSpectrum::PhatFvec::clear()
//~ {
	//~ x.clear();
	//~ y.clear();
	//~ z.clear();
	
	//~ f.clear();
//~ }

//~ ////////////////////////////////////////////////////////////////////////

void PowerSpectrum::PhatFvec::emplace_back(PhatF const& pHatF)
{
	// Assume that pHat is properly normalized
	x.emplace_back(pHatF.pHat.x1);
	y.emplace_back(pHatF.pHat.x2);
	z.emplace_back(pHatF.pHat.x3);
	
	f.emplace_back(pHatF.f);
}

//~ ////////////////////////////////////////////////////////////////////////

//~ PowerSpectrum::PhatFvec PowerSpectrum::PhatFvec::Join
	//~ (PhatFvec&& first, PhatFvec&& second)
//~ {
	//~ // Steal the first data, copy in the second (can't steal both)
	//~ auto retVec = PhatFvec(std::move(first));
	//~ retVec.append(second);
	//~ return retVec;
//~ }

////////////////////////////////////////////////////////////////////////
// ShapedParticleContainer
////////////////////////////////////////////////////////////////////////

const std::shared_ptr<ShapeFunction> PowerSpectrum::ShapedParticleContainer::delta(new h_Delta());

////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer::ShapedParticleContainer
	(std::vector<ShapedJet> const& jets, bool const normalizeF):
PhatFvec(VecPhatF(jets, normalizeF))
{
	for(auto const& jet : jets)
		shapeVec.emplace_back(jet.shape.Clone());
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer::ShapedParticleContainer
	(std::vector<PhatF> const& particles, 
	std::shared_ptr<ShapeFunction> const& theirSharedShape):
PhatFvec()
{
	append(particles, theirSharedShape);
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::ShapedParticleContainer::append
	(std::vector<PhatF> const& tail, 
	std::shared_ptr<ShapeFunction> const& theirSharedShape)
{
	shapeVec.reserve(size() + tail.size());
	PhatFvec::reserve(shapeVec.capacity());
	
	for(PhatF const& particle : tail)
		PhatFvec::emplace_back(particle);
	
	// Keep pushing back the shared shape until all are in place
	while(shapeVec.size() < this->size())
		shapeVec.push_back(theirSharedShape);
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::ShapedParticleContainer::emplace_back(PhatF const& pHatF, 
	std::shared_ptr<ShapeFunction> const& itsShape)
{
	PhatFvec::emplace_back(pHatF);
	shapeVec.push_back(itsShape);
}

////////////////////////////////////////////////////////////////////////
// DetectorObservation
////////////////////////////////////////////////////////////////////////

PowerSpectrum::DetectorObservation::DetectorObservation(std::vector<vec3_t> const& tracks_in, 
	std::vector<vec3_t> const& towers_in, std::vector<real_t> const& towerAreas_in):
tracks(tracks_in), towers(towers_in), towerAreas(towerAreas_in)
{
	NormalizeF();
	CheckValidity_TowerArea();
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::DetectorObservation::pHatMag(size_t const i) const
{
	// This is only used by IsNormalized(), so we arbitrarily assign 
	// the first indices to tracks, and the later indices to towers.
	if(i < tracks.size())
		return tracks.at(i).pHat.Mag();
	else
		return towers.at(i).pHat.Mag();
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::DetectorObservation::CheckValidity_TowerArea() const
{
	if((towers.empty() and towerAreas.size())
		or ((towerAreas.size() not_eq towers.size()) and  (towerAreas.size() not_eq 1)))
		throw std::invalid_argument(std::string("DetectorObservation: there must either be one ")
			+ "fractional area for every track ore one fractional area for all to share");
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::DetectorObservation::NormalizeF_to(real_t const total_f)
{
	tracks.NormalizeF_to(total_f);
	towers.NormalizeF_to(total_f);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::DetectorObservation::fTotal() const
{
	return tracks.fTotal() + towers.fTotal();
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::DetectorObservation::fInner() const 
{
	return tracks.fInner() + towers.fInner();
}

////////////////////////////////////////////////////////////////////////

size_t PowerSpectrum::DetectorObservation::size() const 
{
	return tracks.size() + towers.size();
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::DetectorObservation PowerSpectrum::DetectorObservation::NaiveObservation() const
{	
	DetectorObservation trivial; // blank observation
	
	trivial.tracks = tracks;
	trivial.tracks.insert(trivial.tracks.end(), towers.cbegin(), towers.cend());
	
	return trivial;
}
	
////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer PowerSpectrum::DetectorObservation::MakeExtensive
	(double const f_trackR, double const u_trackR) const
{
	PowerSpectrum::ShapedParticleContainer container;
	
	if(tracks.size())
	{
		
												GCC_IGNORE_PUSH(-Wfloat-equal)
		if((f_trackR == real_t(0)) or (u_trackR == real_t(1)))
			container.append(tracks);
		else
			container.append(tracks,
				ShapeFunction::Make<h_PseudoNormal>(f_trackR * AngularResolution(), u_trackR));
												GCC_IGNORE_POP
	}

	CheckValidity_TowerArea();
	
	// Either there is one area for every tower, or one for all to share
	if(towerAreas.size() < towers.size())
		container.append(towers, 
			ShapeFunction::Make<h_Cap>(towerAreas.front()));
	else
	{
		for(size_t i = 0; i < towers.size(); ++i)
			container.emplace_back(towers[i], ShapeFunction::Make<h_Cap>(towerAreas[i]));
	}
	
	return container;
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::DetectorObservation::SmearedDistance_TrackTower(real_t const r)
{
	// Fit to Monte Carlo integration results
	static constexpr real_t a = 0.667;
	static constexpr real_t b = 2.38;
	
	return std::pow(std::pow(a, b) + std::pow(r, b), real_t(1)/b);	
}

PowerSpectrum::real_t PowerSpectrum::DetectorObservation::SmearedDistance_TowerTower(real_t const r)
{
	// Fit to Monte Carlo integration results
	static constexpr real_t a = 0.641;
	static constexpr real_t b = 2.29;
	
	return std::pow(std::pow(a, b) + std::pow(r, b), real_t(1)/b);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::DetectorObservation::AngularResolution
	(real_t const ff_fraction) const
{
	struct AngleWeight
	{
		real_t angle; 
		real_t weight;
		
		//~ using iter_t = std::vector<AngleWeight>::iterator;
		
		AngleWeight() = default;
		AngleWeight(real_t const angle_in, real_t const weight_in):
			angle(angle_in), weight(weight_in) {}
		
		bool operator<(AngleWeight const& rhs) const
		{
			return (this->angle < rhs.angle);
		}
		
		// Sort a vector whose head is sorted and tail in unsorted
		static void ReSort(std::vector<AngleWeight>& headTail, size_t const n_sorted)
		{
			assert(n_sorted <= headTail.size());
			std::sort(headTail.begin() + n_sorted, headTail.end());
			
			if(n_sorted > 0)
				std::inplace_merge(headTail.begin(), headTail.begin() + n_sorted, headTail.end());
		}
	};	
	
	CheckValidity_TowerArea();
	if(std::fabs(1. - this->fTotal()) > 1e-8)
		printf("wtf: %.3e\n", std::fabs(1. - this->fTotal()));
	
	if(size() < 3)
	{
		if(size() == 2) // Only two tracks? The angular resolution is approximately M_Pi
			return M_PI;
		else
			return INFINITY;
	}
	else
	{
		std::vector<real_t> twrRadii;
		
		size_t reductions_trk = 0;
		size_t reductions_twr = 0;		
		
		// Integrating \int_0^thetaR dOmega = 
		// 2 pi (1-cos(thetaR)) = Omega ==> sin(thetaR/2)**2 = surfaceFraction
		for(real_t const surfaceFraction : towerAreas)
			twrRadii.emplace_back(real_t(2)*std::asin(std::sqrt(surfaceFraction)));
		
		// If all towers share the same radii, simply copy that universal radii
		// (this rote copying simplifies the logic of the inner loop)
		if(twrRadii.size() < towers.size())
			twrRadii.assign(towers.size(), twrRadii.at(0));
			
		// Create a list of inter-particle angles and correlation weight = f_i * f_j
		std::vector<AngleWeight> angleVec;
		angleVec.reserve(this->size());
		size_t n_sorted = 0;
		
		// We expect that the angular resolution will be calculated by 
		// N inter-particle angles (Ex(<f|f>) ~= 1/n, and Ex(f) = 1/n, 
		// so n * f * f ~= <f|f>
		// Even though we only store a fraction of the total inter-particle angles, 
		// we know they are the smallest \p maxSize inter-particle angles
		// because the only way you can evicted from the smallest list
		// is if something is smaller than you.
		for(size_t const angleVec_maxSize : {size_t(2. * ff_fraction * double(this->size())),
			kdp::GaussSum(this->size() - 1)})
		{
			for(auto trk = tracks.cbegin(); trk not_eq tracks.cend(); ++trk)
			{
				// We only calculate the lower-half of the symmetric inter-particle matrix
				for(auto trk_other = tracks.cbegin(); trk_other not_eq trk; ++trk_other)
				{
					angleVec.emplace_back(
						// No smearing between tracks; assume very well measured
						trk->pHat.InteriorAngle(trk_other->pHat),
							trk->f * trk_other->f);
				}
				
				for(size_t twr_i = 0; twr_i < towers.size(); ++twr_i)
				{
					// We calculate the dimensionless smeared angle, then rescale it by the tower radius
					angleVec.emplace_back(
						twrRadii[twr_i] * SmearedDistance_TrackTower(
							trk->pHat.InteriorAngle(towers[twr_i].pHat)/twrRadii[twr_i]),
						trk->f * towers[twr_i].f);
				}
				
				if(angleVec.size() > 2 * angleVec_maxSize)
				{
					++reductions_trk;
					AngleWeight::ReSort(angleVec, n_sorted);
					angleVec.resize(angleVec_maxSize);
					n_sorted = angleVec_maxSize;
				}
			}
			
			for(size_t twr_i = 0; twr_i < towers.size(); ++twr_i)
			{
				// We only calculate the lower-half of the symmetric inter-particle matrix
				for(size_t twr_j = 0; twr_j < twr_i; ++twr_j)
				{
					// For two towers, we use their shared radius
					real_t const twoTowerRadius = std::hypot(twrRadii[twr_i], twrRadii[twr_j]);
					
					angleVec.emplace_back(
						twoTowerRadius	* SmearedDistance_TowerTower(
							towers[twr_i].pHat.InteriorAngle(towers[twr_j].pHat)/twoTowerRadius),
						towers[twr_i].f * towers[twr_j].f);
				}
				
				if(angleVec.size() > 2 * angleVec_maxSize)
				{
					++reductions_twr;
					AngleWeight::ReSort(angleVec, n_sorted);
					angleVec.resize(angleVec_maxSize);
					n_sorted = angleVec_maxSize;
				}
			}
			// Now sort angles from smallest to largest; keeping the weight 
			// properly attached to its corresponding angle is why we needed the struct
			//~ std::sort(angleVec.begin(), angleVec.end());
			// We only want to sort by angle (first), so we define a temporary lambda expression
			//~ [](AngleWeight const& left, AngleWeight const& right)
			//~ {return left.first < right.first;});
			AngleWeight::ReSort(angleVec, n_sorted);			
		
			real_t weight = real_t(0);
			/* Now we add up the smallest angles until their collective weight 
			 * exceeds some weight target (as a function of <f|f>, the approximate
			 * height of the power spectrum's asymptotic plateau).
			 * We scale weight_target by 1/2 because each angle/weight appears 
			 * twice in the full matrix, but we only calculated the lower half
			*/ 
			real_t const weight_target = 
				std::min(real_t(0.5)*(1. - fInner()), real_t(0.5) * ff_fraction * fInner()); 
			// We use the geometric mean because it averages over scales
			real_t geoMean = real_t(0);
			
			size_t i = 0;
			
			for(; (i < angleVec.size()) and (weight < weight_target); ++i)
			{
				//~ geoMean += angleVec[i].second * std::log(angleVec[i].first);
				//~ weight += angleVec[i].second;
				geoMean += angleVec[i].weight * std::log(angleVec[i].angle);
				weight += angleVec[i].weight;
			}
			
			// The loop aborted because we ran out of angles
			if(i == angleVec.size())
			{
				angleVec.clear();
				n_sorted = 0;
				printf("	%.3e  %.3e  %.3e  %lu\n", weight, fInner(), weight_target, this->size());
				std:: cout << "retrying" << std::endl;
				continue;
			}
			
			printf("[%lu, %lu]", reductions_trk, reductions_twr);
			
			return std::exp(geoMean / weight);
		}
		
		// This should never happen; the second iteration should be the safety net
		assert(false);
		return -INFINITY;
	}
}

////////////////////////////////////////////////////////////////////////
// PowerSpectrum::Job
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
	// nextTile is a std::atomic, so assigning tiles based on its incremented value is thread-safe.
	// It would not be thread safe to exit a condition_variable based upon tileIndex; 
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

void PowerSpectrum::Job::Add_Hl(std::vector<real_t>& Hl_partial, size_t const numTiles)
{
	// Don't waste time synchronizing if no tiles were pulled; nothing to do
	if(numTiles) 
	{
		//Synchronize Hl_total and remainingTiles (job completion)
		std::unique_lock<std::mutex> lock(jobLock);

		if(Hl_total.empty())
			// The first thread to finish simply sets it's data (std::move for quickness)
			Hl_total = std::move(Hl_partial);
		else
			Hl_total += Hl_partial; // Otherwise we add to the existing
			
		remainingTiles -= numTiles;
		assert(remainingTiles <= tileVec.size());
		
		// Unlock before notify to prevent hurry-up-and-wait
		lock.unlock();
	
		// If all tiles are done, notify the sub-manager (we are outside of lock, 
		// so this could create a race to notify, but that's not a problem).
		if(remainingTiles == 0)
		{
			// Sanity check; if the job is done, all tiles were assigned
			assert(nextTile >= tileVec.size());
			
			done.notify_one();
		}
	}
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Job::Get_Hl()
{
	// Synchronize remainingTiles (job completion)
	std::unique_lock<std::mutex> lock(jobLock);

	// Wait until there are no more threads actively working on this job.
	while(remainingTiles)
		done.wait(lock);
	
	// When the job is done, steal the final vector and return
	return std::move(Hl_total);
}

////////////////////////////////////////////////////////////////////////
//  PowerSpectrum
////////////////////////////////////////////////////////////////////////
		
void PowerSpectrum::Hl_Thread()
{
	// We calculate the entire square tile at once;
	static constexpr size_t incrementSize = kdp::Squared(tileWidth);
	using array_t = std::array<real_t, incrementSize>;
	using shapePtr_t = std::shared_ptr<ShapeFunction>;
				
	// Permanently store the arrays which are vectorized, for better compiler optimization	
	// Recursively calculate Pl(pHat_i.Dot(pHat_j))
	RecursiveLegendre_Increment<array_t> Pl_computer; // Pl_computer.z = pHat_i.Dot(pHat_j)
	array_t fProd; // f_i * f_j (not l-dependent)
	array_t Hl_accumulate;
	// The values of hl for each column (the inner loop); we do not need one for rows
	std::array<real_t, tileWidth> colShapeVal; 
			
	std::shared_ptr<Job> job;
		
	// Request jobs; Dispatch() will block until there are jobs, 
	// and return nullptr when it's time to die (exit the main loop and return)
	while((job = Dispatch()))
	{
		size_t const lMax = job->lMax; // Tell compiler that this is a constant
		bool const symmetric = (job->left == job->right); // A symmetric outer-product
		
		// Each tile adds to this, so we must zero-initialize
		std::vector<real_t> hl_partial(lMax, real_t(0));
		
		TileSpecs tile;
		size_t tileCount = 0; // How many tiles do we calculate
		
		// Get the index of our next tile. When GetTile() returns false,
		// there are no more tiles to analyze in this job
		while(job->GetTile(tile))
		{
			++tileCount;
			
			// Development assertions
			
			// No width should exceed tileWidth
			assert(tile.row_width <= tileWidth);
			assert(tile.col_width <= tileWidth);
			// Unless it's the FINAL tile, the row should always be full of columns.
			if(tile.col_width < tileWidth) assert(tile.type == TileType::FINAL);
			// We should only get a RIGHT edge for asymmetric outer products
			if(tile.type == TileType::RIGHT)	assert(not symmetric);
			if(symmetric and (tile.type == TileType::DIAGONAL))
			{
				assert(tile.row_beg == tile.col_beg);
				assert(tile.row_width = tile.col_width);
			}
			
			/* In general, "left" supplies the rows and "right" the columns.
			 * For the RIGHT edge of the asymmetric product, 
			 * the columns are full but the rows are not.
			 * This creates an SIMD problem, because the vectorized loop 
			 * is the inner/column loop, which is now not full.
			 * We can fix this problem by redefining which vector supplies rows and columns.
			 * This is simpler than creating a separate loop for RIGHT tiles, 
			 * with its own control logic.
			 * WARNING: We assume that this transposition was anticipated 
			 * when the tile was defined, so that row/col beg/width were already swapped.
			 * Since left/right are not kept in tileSpecs, we must swap them here.
			*/ 
			ShapedParticleContainer const* const rows = (tile.type == TileType::RIGHT) ? job->right : job->left;
			ShapedParticleContainer const* const cols = (tile.type == TileType::RIGHT) ? job->left : job->right;
			
			/* ShapeParticleContainer.shapeVec is a vector of pointers to shape functions.
			 * If all the particles use the same shape function, it is the same pointer over and over.
			 * This is beneficial because ShapeFunction.hl operates recursively, 
			 * and remembers its last value. So the first time hl(10) is called, 
			 * math is done, but the second time hl(10) is called, we look up the cached value.
			 * But since ShapeFunction's are not thread-safe, we must clone any we intend to use.
			 * But if we have repeated pointers in shapeVec, a simple cloning 
			 * will create a unique clone each time. We therefore use a std::map 
			 * to map unique shapes to unqiue clones.
			*/ 
			
			// Cloned replicas of row->shapeVec and col->shapeVec
			std::vector<shapePtr_t> colShape, rowShape;
			
			{
				// Map the original column shapes to clones
				std::map<shapePtr_t, shapePtr_t> cloneMap;
				
				// First collect all unique shapes by using cloneMap as a std::set
				// (mapping each unique key to the nullptr placeholder)
				for(size_t j = 0; j < tile.col_width; ++j)
					cloneMap[cols->shapeVec.at(tile.col_beg + j)] = shapePtr_t(nullptr);
				
				// Now map all unique shapes to clones
				for(auto shape_it = cloneMap.begin(); shape_it not_eq cloneMap.end(); ++shape_it)
					shape_it->second = shape_it->first->Clone();
					
				if(cloneMap.size() > 1) // We replicate cols->shapeVec with the clones. 
				{
					// Use at() as sanity check, because it throws an exception if the pointer is not found
					for(size_t j = 0; j < tile.col_width; ++j)
						colShape.push_back(cloneMap.at(cols->shapeVec.at(tile.col_beg + j)));
				}
				else // Otherwise there is only one unique shape, and it's the only one we need.
					colShape.push_back(cloneMap.begin()->second);
			}
			
			// If every column uses the same shape, we only need colShape.front()
			bool const colShapeInhomogeneous = (colShape.size() == tile.col_width);
			
			if(symmetric and (tile.type == TileType::DIAGONAL))
				rowShape = colShape; // Rows are the same as columns, so shapes are the same
			else
			{
				// We do pretty much the same thing as the previous, colShape scope
				std::map<shapePtr_t, shapePtr_t> cloneMap;
				
				for(size_t i = 0; i < tile.row_width; ++i)
					cloneMap[rows->shapeVec.at(tile.row_beg + i)] = shapePtr_t(nullptr);
				
				for(auto shape_it = cloneMap.begin(); shape_it not_eq cloneMap.end(); ++shape_it)
					shape_it->second = shape_it->first->Clone();
					
				if(cloneMap.size() > 1)
				{
					for(size_t i = 0; i < tile.row_width; ++i)
						rowShape.push_back(cloneMap.at(rows->shapeVec.at(tile.row_beg + i)));
				}
				else
					rowShape.push_back(cloneMap.begin()->second);
			}
			bool const rowShapeInhomogeneous = (rowShape.size() == tile.row_width);
			
			/* If all the rows share the same shape, and all the columns have the same shape: 
			 * 	H_l = h_l^{row} * h_l^{col} * <f_row| P_l( |p_row> <p_col| ) |f_col>
			 * In this case, we can multiply the shape after the accumulating Hl_accumulate 
			 * (which uses less FLOPS). Otherwise, we must multiply shape before we accumulate.
			*/
			bool const shapeBeforeAccumulate = ((rowShape.size() > 1) or colShapeInhomogeneous);
			
			///////////////////////////////////////////////////////////////
			// Prepare for the l-loop by calculating inter-particle weights and dot-products
			
			/* To efficiently calculating a small, symmetric outer product
			 * (i.e. Hl_Jet for N=3 jets), the last tile of a symmetric outer is "ragged".
			 * This means that we only calculate the lower half and diagonal.
			 * Doing this creates a *small* (2%) speed hit for large outer products, 
			 * but it is definitely worth it (50% faster) for very small outer products.
			 * As N approaches tileWidth, ragged tiles are definitely slower,
			 * so we only use the ragged tile when less than half the tile is full.
			*/ 
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
			// so any necessary zero-buffer will remain zero.
			Pl_computer.z.fill(real_t(0));
			fProd.fill(real_t(0));
			Hl_accumulate.fill(real_t(0));
			colShapeVal.fill(real_t(0));
				
			/* Fill fProd:
			 * Only FINAL tiles have partially full rows, so we can hard-code
			 * tileWidth as the j-loop end-condition for non-FINAL tiles.
			 * Symmetry factors accounting for un-computed tiles are also applied here
			 * (to double the contribution from off-diaonal tiles, simply double their f).
			 * Testing reveals that three loops is actually noticeably faster,
			 * which motivates the less readable code.
			*/			
			if(tile.type == TileType::FINAL)
			{
				if(ragged)
				{
					size_t k = 0;
					
					for(size_t i = 0; i < tile.row_width; ++i)
					{
						for(size_t j = 0; j <= i; ++j)
						{
							// Only off-diagonal elements need doubling
							real_t const symmetry = (j == i) ? real_t(1) : real_t(2);
							
							fProd[k++] = symmetry * 
								(rows->f[tile.row_beg + i] * cols->f[tile.col_beg + j]);
						}
					}
					
					// Development assertions
					assert(k == kdp::GaussSum(tile.row_width));
					assert(k <= k_max);
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
				// DIAGONAL tiles which are full; they are the only tiles here which don't need doubling.
				
				real_t const symmetry = (symmetric and (tile.type not_eq TileType::DIAGONAL)) ? 
					real_t(2) : real_t(1);
				
				for(size_t i = 0; i < tile.row_width; ++i)
					for(size_t j = 0 ; j < tileWidth; ++j) // Hard-code tileWidth for speed
						fProd[i * tileWidth + j] = symmetry * 
							(rows->f[tile.row_beg + i] * cols->f[tile.col_beg + j]);
			}
			
			// Fill pDot with the dot product of the 3-vectors:
			// Testing reveals that it is faster to do x, y, and z in separate loops
			// (since this caches all of x, then all of y, then all of z)
			// Testing also shows that 3 versions is faster, 
			// even though it makes a terrible mess of the code.
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
						for(size_t j = 0 ; j < tileWidth; ++j) // Hard-code tileWidth for speed
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
			Pl_computer.Reset(k_max);
			
			for(size_t l = 1; l <= lMax; ++l)
			{
				// The branch is quite predictable, and saves one unused call to Pl_computer.Next()
				// (i.e. if we put the call to Pl_computer.Next() at the end of the loop).
				// The speed advantage is actually noticeable.
				if(l > 1)
					Pl_computer.Next();
				assert(Pl_computer.l() == l);
				
				// Do the original calculation for delta-distribution particles
				for(size_t k = 0; k < k_max; ++k)
					Hl_accumulate[k] = Pl_computer.P_l()[k] * fProd[k];
				
				// Do we need to handle shape before we accumulate?
				if(shapeBeforeAccumulate)
				{
					// If columns have different shapes, cache the value of each shape function.
					// Redundant calls to repeated shapes are still somewhat efficient,
					// as the cached value is returned (instead of recalculating).
					if(colShapeInhomogeneous)
					{
						for(size_t j = 0; j < tile.col_width; ++j)
							colShapeVal[j] = colShape[j]->hl(l);
					}
					else
						colShapeVal.fill(colShape.front()->hl(l)); // All the same shape, fill
					
					// The row's h_l will be cached one at a time, since rows are the outer loop
					if(ragged)
					{
						size_t k = 0;
						for(size_t i = 0; i < tile.row_width; ++i)
						{
							real_t const rowShapeVal = 
								(rowShapeInhomogeneous ? rowShape[i] : rowShape.front())->hl(l);
							
							for(size_t j = 0; j <= i; ++j)
								Hl_accumulate[k++] *= rowShapeVal * colShapeVal[j];
						}
					}
					else
					{
						for(size_t i = 0; i < tile.row_width; ++i)
						{
							real_t const rowShapeVal = 
								(rowShapeInhomogeneous ? rowShape[i] : rowShape.front())->hl(l);
							
							/* Wait! If this is a final tile, it's row may not be full!
							 * Doesn't matter, we *= garbage into something 
							 * which is already zero (because fProd is zero there).
							 * Hardcoding tileWidth  is faster because MOST tiles are full,
							 * and this solution creates less branches overall.
							*/ 
							for(size_t j = 0 ; j < tile.col_width; ++j)
							{
								Hl_accumulate[i * tileWidth + j] *= rowShapeVal * colShapeVal[j];
								
																					//~ GCC_IGNORE_PUSH(-Wfloat-equal)
								//~ if(j >= tile.col_width) // Check that assumption
									//~ assert(Hl_accumulate[i * tileWidth + j] == real_t(0));
																					//~ GCC_IGNORE_POP
							}
						}
					}
				}
				
				// Now that we've applied shape, we can accumulate all the terms
				real_t Hl_sum = kdp::BinaryAccumulate_Destructive(Hl_accumulate, sumSize);
				
				// If we didn't handle shape yet, now is the time
				if(not shapeBeforeAccumulate)
					Hl_sum *= rowShape.front()->hl(l) * colShape.front()->hl(l);
				
				hl_partial[l - 1] += Hl_sum; // l = 0 is not stored
			}// end l-loop
		}
			
		// No more tiles; this thread is done doing major work.
		// Since it has nothing better to do, use the thread to 
		// add to Hl_total before requesting the next job.
		job->Add_Hl(hl_partial, tileCount);
	}
}

////////////////////////////////////////////////////////////////////////

std::shared_ptr<PowerSpectrum::Job> PowerSpectrum::Dispatch()
{
	// Synchronize keepAlive and jobQueue
	std::unique_lock<std::mutex> dispatch(dispatch_lock);
	
	/* Some will say while(true) is a bad practice; a never-ending loop?
	 * However, I feel this is the most readable way to say what this loop does.
	 * In perpetuity:
	 *    1. Look for jobs to dispatch, pruning those which are already fully assigned.
	 *    2. If there are no jobs and we are supposed to keepAlive, sleep. 
	 *    3. Only when there are no jobs and it's time to die do we
	 *       kill the calling thread by returning nullptr.
	*/
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
				jobQueue.pop_front(); // Remember, we hold the dispatch_lock, so this is thread-safe
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
	// keepAlive is guarded by both locks
	std::unique_lock<std::mutex> threadSafe(threadSafe_lock);
	std::unique_lock<std::mutex> dispatch(dispatch_lock);
	
	keepAlive = false; // Stop all threads after all current jobs are complete
	dispatch.unlock(); // Unlock dispatch so threads can finish existing jobs
		
	newJob.notify_all(); // Wake up any sleeping threads to finish existing jobs
	
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
		// but then the code becomes less readable. No time is spent here,
		// so we choose readability over compactness
		if(left == right)
		{
			for(size_t i = 0; i < left->size(); i += tileWidth) // left is for the rows
			{
				// Detect the last row of tiles
				TileType rowType_base = ((i + tileWidth) >= left->size()) ? 
					TileType::BOTTOM : TileType::CENTRAL;
				
				// Only complete the lower-half of the outer product
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
					{
						// Swap rows and columns so rows are full
						tileVec.push_back(TileSpecs{j, col_width, i, row_width, type});
						assert(tileVec.back().row_width == tileWidth);
					}
					else
						tileVec.push_back(TileSpecs{i, row_width, j, col_width, type});
				}
			}
		}
		
		// We should have already caught the three conditions that cause zero tiles
		assert(tileVec.size() > 0);
		
		std::shared_ptr<Job> job = std::make_shared<Job>(left, right, lMax, std::move(tileVec));
		
		// Lock dispatch to add the job to the queue
		{
			std::lock_guard<std::mutex> dispatch(dispatch_lock);
				jobQueue.push_back(job);
		}
		// Unlock before notifying so that work can begin immediately (no hurry-up-and-wait)
		
		// Wake up the threads to do their work.
		if(job->RemainingTiles() <= minTilesPerThread)
			newJob.notify_one();
		else
			newJob.notify_all();
			
		std::vector<real_t> Hl_final = job->Get_Hl(); // Wait for job to finish
		
		// Lock threadSafe to notify completion of this job.
		threadSafe.lock();
			--activeJobs;
		
		/* Normally we would now unlock before notification, 
		 * to prevent a hurry-up-and-wait (i.e. we notify while still holding
		 * the mutex, so that threads cannot wakeup).
		 * But this time, the only thing that's waiting for jobDone is the 
		 * dtor waiting for active jobs to finish.
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
	ShapedParticleContainer const& jets, std::vector<real_t> const& hl_detector_Filter)
{
	std::vector<real_t> Hl_vec = Hl_Job(&jets, &jets, lMax);
	
	if(hl_detector_Filter.size())
	{
		if(hl_detector_Filter.size() < lMax)
			throw std::runtime_error("PowerSpectrum::Hl_Jet: on-axis filter too short");
	 	
		for(size_t lMinus1 = 0; lMinus1 < Hl_vec.size(); ++lMinus1)
			Hl_vec[lMinus1] *= kdp::Squared(hl_detector_Filter[lMinus1]);
	}
		
	return Hl_vec;
}

////////////////////////////////////////////////////////////////////////
	
std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Hybrid(size_t const lMax,
	ShapedParticleContainer const& jets, std::vector<real_t> const& hl_detector_Filter,
	ShapedParticleContainer const& particles,
	std::vector<real_t> const& Hl_Obs_in)
{
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
	
	// Since this Hl is composed from 2 (or 3) separate power spectra,
	// we can start those Hl calculations asynchronously, to better utilize the thread pool.
	std::future<std::vector<real_t>> Hl_Obs_future, Hl_jets_particles_future, Hl_jets_future;
	
	bool const recalculate_Obs = (Hl_Obs_in.size() < lMax);
	
	// Reuse Hl_Obs if possible, but if not launch the particles-particles job
	// This is probably the longest job, so we start it first to warm up the thread pool
	// (and so we construct the jet's ShapedParticleContainer while something else is running).
	if(recalculate_Obs)
	{
		Hl_Obs_future = std::async(std::launch::async, &PowerSpectrum::Hl_Obs, 
			this, lMax, std::cref(particles));
		// Before there were two Hl_Obs, and he had to help the compiler disentangle the ambiguity
		// https://stackoverflow.com/questions/27033386/stdasync-with-overloaded-functions
		// Keep this around as an example of how to do that when using std::async		
		//~ static_cast<std::vector<real_t>(PowerSpectrum::*)(size_t const,
			//~ ShapedParticleContainer const&)>(&PowerSpectrum::Hl_Obs), this, 
			//~ lMax, std::cref(particles));
	}
	else
		Hl.assign(Hl_Obs_in.cbegin(), Hl_Obs_in.cbegin() + lMax);	
	
	// Start the calculations involving the jets
	{
		Hl_jets_particles_future = std::async(std::launch::async, &PowerSpectrum::Hl_Job, this,
			&jets, &particles, lMax);
			
		Hl_jets_future = std::async(std::launch::async, &PowerSpectrum::Hl_Jet, this,
			lMax, std::cref(jets), std::cref(hl_detector_Filter));
			
		//~ static_cast<std::vector<real_t>(PowerSpectrum::*)(size_t const,
		//~ ShapedParticleContainer const&, std::vector<real_t> const&)>(&PowerSpectrum::Hl_Jet), this,
		//~ lMax, std::cref(jets), std::cref(hl_detector_Filter));
		
		// With all jobs in the queue, get the one that should come off first.
		if(recalculate_Obs)
			Hl = Hl_Obs_future.get();
			
		{
			// Assign this to a temporary spot so we can apply the detector filter
			auto Hl_jets_particles = Hl_jets_particles_future.get();
			
			if(hl_detector_Filter.size())
			{
				if(hl_detector_Filter.size() < lMax)
					throw std::runtime_error("PowerSpectrum::Hl_Jet: on-axis filter too short");
					
				// Add the doubling factor when we apply the detector filter
				for(size_t lMinus1 = 0; lMinus1 < Hl_jets_particles.size(); ++lMinus1)
					Hl_jets_particles[lMinus1] *= real_t(2) * hl_detector_Filter[lMinus1];
			}
			else
				Hl_jets_particles *= real_t(2);
			
			Hl += Hl_jets_particles;
		}
		
		Hl += Hl_jets_future.get();
	}
			
	Hl *= real_t(0.25); // To reduce FLOPS, divide by four once
	return Hl;
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::WriteToFile(std::string const& filePath,
	std::vector<std::vector<real_t>> const& Hl_set, std::string const& header)
{
	std::ofstream file(filePath, std::ios::trunc);
	
	file << "#" << header << "\n"; // Add the header as a comment line
	
	// Find the largest l in Hl_set. Use a lambda to const-init lMax
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
		std::vector<char> buff_vec;
		buff_vec.reserve((Hl_set.size() + 1) * 30);
		char* buff = buff_vec.data();

		sprintf(buff, l_format, 0lu);
		file << buff;		
		
		// H_0 always equals 1
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
				// Use a nonsense value of -1 when lMinus1 exceeds the length of the set
				real_t const val = (Hl_set[k].size() > lMinus1) ? Hl_set[k][lMinus1] : -1.;
				sprintf(buff, Hl_format, val);
				file << buff;
			}
			
			file << "\n";
		}
	}
}

////////////////////////////////////////////////////////////////////////

std::pair<std::vector<PowerSpectrum::real_t>, std::vector<std::vector<PowerSpectrum::real_t>>>
PowerSpectrum::AngularCorrelation(std::vector<std::vector<real_t>> const& Hl_set, 
	size_t const zSamples)
{
	using vec_t = std::vector<real_t>;
	RecursiveLegendre_Increment<vec_t> Pl_computer;

	for(size_t i = 0; i < zSamples; ++i)
		Pl_computer.z.emplace_back(real_t(-1) + ((real_t(i) + real_t(0.5)) * real_t(2)) / real_t(zSamples));
	assert(Pl_computer.z.size() == zSamples);
	
	// Accumulate A(z) for each Hl, default emplacing l=0	
	std::vector<vec_t> A_vec(Hl_set.size(), vec_t(zSamples, real_t(1)));
	
	size_t const lMax = [&]()
	{
		std::vector<size_t> lMax_vec;
		
		for(auto const& Hl : Hl_set)
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
			
			for(size_t i = 0; i < Hl_set.size(); ++i)
			{
				vec_t const& Hl = Hl_set[i];
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

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Write_AngularCorrelation(std::string const& filePath, 
	std::vector<std::vector<real_t>> const& Hl_set, size_t const zSamples, 
	std::string const& header)
{
	std::ofstream file(filePath, std::ios::trunc);
	
	constexpr char const* data_format = "%.16e ";
	
	std::vector<char> buff_vec((Hl_set.size() + 1) * strlen(data_format));
	char* buff = buff_vec.data();
	
	if(not file.is_open())
		throw std::ofstream::failure("Cannot open: <" + filePath + "> for writing");
	else
	{
		file << "# " << header << "\n"; // Add the header as a comment line
		
		// Get the angular correlation series
		auto const series = AngularCorrelation(Hl_set, zSamples);
		
		for(size_t i = 0; i < series.first.size(); ++i)
		{
			// First column is the z-values
			sprintf(buff, data_format, series.first[i]);
			file << buff;
			
			// Remaining columns are A(z) for the given Hl_set
			for(size_t k = 0; k < Hl_set.size(); ++k)
			{
				sprintf(buff, data_format, series.second[k][i]);
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
