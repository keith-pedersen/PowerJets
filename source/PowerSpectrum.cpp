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
	(real_t const angularResolution, real_t const f_trackR, double const u_trackR) const
{
	PowerSpectrum::ShapedParticleContainer container;
	assert(kdp::AbsRelError(this->fTotal(), real_t(1)) < 1e-8);
	
	if(tracks.size())
	{	
												GCC_IGNORE_PUSH(-Wfloat-equal)
		if((f_trackR == real_t(0)) or (u_trackR == real_t(1)))
			container.append(tracks);
		else
			container.append(tracks,
				ShapeFunction::Make<h_PseudoNormal>(f_trackR * angularResolution, u_trackR));
												GCC_IGNORE_POP
	}

	CheckValidity_TowerArea();
	
	// Use the angular resolution to define the minimum tower area
	real_t const towerArea_min = kdp::Squared(std::sin(real_t(0.5) * angularResolution));
	
	// Either there is one area for every tower, or one for all to share
	if(towerAreas.size() < towers.size())
		container.append(towers, 
			ShapeFunction::Make<h_Cap>(std::max(towerAreas.front(), towerArea_min)));
	else
	{
		for(size_t i = 0; i < towers.size(); ++i)
		{
			container.emplace_back(towers[i], 
				ShapeFunction::Make<h_Cap>(std::max(towerArea_min, towerAreas[i])));
		}
	}
	
	return container;
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::SmearedDistance_TrackTower(real_t const r)
{
	// Fit to Monte Carlo integration results
	static constexpr real_t a = 0.667;
	static constexpr real_t b = 2.38;
	
	return std::pow(std::pow(a, b) + std::pow(r, b), real_t(1)/b);	
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::SmearedDistance_TowerTower(real_t const r)
{
	// Fit to Monte Carlo integration results
	static constexpr real_t a = 0.641;
	static constexpr real_t b = 2.29;
	
	return std::pow(std::pow(a, b) + std::pow(r, b), real_t(1)/b);
}

////////////////////////////////////////////////////////////////////////
// PowerSpectrum::Job
////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job::Job():
	remainingTiles(0), nextTile(0) {}
	
////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job::TileSpecs* PowerSpectrum::Job::NextTile()
{
	// nextTile is a std::atomic, so assigning tiles based on its incremented value is thread-safe.
	// It would not be thread safe to exit a condition_variable based upon tileIndex; 
	// however, job completion is based on tiles which are complete, not merely assigned.
	size_t const tileIndex = nextTile++;
	
	return (tileIndex < tileVec.size()) ? 
		tileVec[tileIndex].operator->() : nullptr;
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job::WaitTillDone() const
{
	// Synchronize remainingTiles (job completion)
	std::unique_lock<std::mutex> lock(jobLock);

	// Wait until there are no more threads actively working on this job.
	while(remainingTiles)
		done.wait(lock);
}

////////////////////////////////////////////////////////////////////////
// PowerSpectrum::Job_Hl
////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job_Hl::Job_Hl(size_t const lMax_in, 
	ShapedParticleContainer const& left,
	ShapedParticleContainer const& right):
Job(),
rows(&left), cols(&right), lMax(lMax_in)
{
	if(rows == cols)
		tileVec = TileSpecs::Partition_Symmetric_ptr(rows->size());
	else
		tileVec = TileSpecs::Partition_ptr(rows->size(), cols->size());
		
	remainingTiles = tileVec.size();
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job_Hl::Accumulate::operator()(std::vector<real_t>& Hl_partial)
{
	if(empty())
		// The first thread to finish simply sets it's data (std::move for quickness)
		this->std::vector<real_t>::operator=(std::move(Hl_partial));
	else
		static_cast<std::vector<real_t>&>(*this) += Hl_partial; // Otherwise we add to the existing
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job_Hl::Finalize_Thread(size_t const numTiles, 
	std::vector<real_t>& Hl_partial)
{
	Job::Finalize_Thread(numTiles, Hl_total, Hl_partial);
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Job_Hl::Get_Hl()
{
	WaitTillDone();
	
	// When the job is done, steal the final vector and return
	return std::move(Hl_total);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job_Hl::ApplyShapes PowerSpectrum::Job_Hl::BitAnd
	(ApplyShapes const left, ApplyShapes const right)
{
	return ApplyShapes(ApplyShapes_t(left) bitand ApplyShapes_t(right));
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job_Hl::ApplyShapes PowerSpectrum::Job_Hl::BitOr
	(ApplyShapes const left, ApplyShapes const right)
{
	return ApplyShapes(ApplyShapes_t(left) bitor ApplyShapes_t(right));
}

////////////////////////////////////////////////////////////////////////

bool PowerSpectrum::Job_Hl::IsBefore(ApplyShapes const val)
{
	return bool(ApplyShapes_t(val));
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::Job_Hl::shapePtr_t> PowerSpectrum::Job_Hl::CloneShapes
	(std::vector<shapePtr_t> const& shapeVec, size_t const begin, size_t const size)
{
	std::vector<shapePtr_t> clones;
	
	if(size)
	{
		std::map<shapePtr_t, shapePtr_t> cloneMap; // map originals to clones
		
		// Collect all unique shapes in shapeVec by first using cloneMap as a std::set
		// (mapping each unique key to the nullptr placeholder)
		for(size_t i = 0; i < size; ++i)
			cloneMap[shapeVec.at(begin + i)] = shapePtr_t(nullptr);
		
		// Now map all unique shapes to clones (replacing the nullptr placeholder)
		for(auto shape_it = cloneMap.begin(); shape_it not_eq cloneMap.end(); ++shape_it)
			shape_it->second = shape_it->first->Clone();
			
		if(cloneMap.size() > 1) // We replicate cols->shapeVec with the clones.
		{
			// Use at() as sanity check, because it throws an exception if the pointer is not found
			for(size_t i = 0; i < size; ++i)
				clones.push_back(cloneMap.at(shapeVec.at(begin + i)));			
		}
		else // Otherwise there is only one unique shape, and it's the only one we need.
			clones.push_back(cloneMap.begin()->second);
	}
	return clones;
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job_Hl::DoTiles()
{
	using shapePtr_t = std::shared_ptr<ShapeFunction>;
	using namespace TiledOuter;
	
	// Each tile adds to this, so we must zero-initialize
	std::vector<real_t> hl_partial(lMax, real_t(0));
	
	size_t tileCount = 0; // How many tiles do we calculate?
	{
		RecursiveLegendre_Increment<array_t> Pl_computer;
		array_t fProd; // f_i * f_j (not l-dependent)
		array_t Hl_accumulate; // Where we accumulate the partial Hl for each tile
		
		// The values of hl for each row and column (to be filled inside the l-loop)
		std::array<real_t, tileWidth> rowShape_l, colShape_l;
		
		bool const symmetric = (rows == cols); // Is the outer-product symmetric?
		TileSpecs* tile; // We request tiles from NextTile() and store them here
		
		while((tile = NextTile())) // NextTile() returns nullptr when they're gone
		{
			assert((tile->num_rows > 0) and (tile->num_cols > 0));
			
			++tileCount;
			bool const diagonal = (symmetric and tile->isDiagonal());
			// Previously we used an empty colShape to indicate a diagonal tile, 
			// but an explicit bool is FAR more readable.
			
			///////////////////////////////////////////////////////////////
			// Clone shape functions before the l-loop
			///////////////////////////////////////////////////////////////
						
			// Clone replicas of row->shapeVec
			std::vector<shapePtr_t> const rowShape = CloneShapes(rows->shapeVec, 
				tile->row_beg, tile->num_rows);
			
			assert((rowShape.size() == tile->num_rows) or (rowShape.size() == 1));
			
			// A diagonal tile (which only exists in symmetric outer-products)
			// has the same shape in both rows and cols, so colShape is redundant.
			// In this case, we don't waste time copying
			std::vector<shapePtr_t> const colShape = diagonal ? 
				std::vector<shapePtr_t>() : 
				CloneShapes(cols->shapeVec, tile->col_beg, tile->num_cols);
				
			if(not diagonal)
				assert((colShape.size() == tile->num_cols) or (colShape.size() == 1));
			
			// Now that we know what shapes we have, we can determine
			// when they will be applied to Hl -- before or after accumulation.
			ApplyShapes const applyShapes = [&]()
			{
				/* We can apply shapes after if there is only 1 shape, 
				 * because the accumulation does not lose any information
				 * (shape * {1, 2, 3, 4, 5, 6} = shape * 21. We check for (rowShape.size == 1), 
				 * rather than (rowShape.size < tile->num_rows), because if 
				 * (num_rows == 1), the distributive argument still applies.
				*/ 
				if(diagonal)
				{
				   // IF: there is only one member of rowShape, it means that
				   // all rows (and cols, in this diagonal tile) have the same shape ...
				   return ((rowShape.size() == 1) ? 
						ApplyShapes::after : // THEN: apply shape after accumulation
						ApplyShapes::bothBefore); // ELSE: apply shape before accumulation
				}
				else
				{
					// Otherwise, we apply the same IF-THEN_ELSE logic as above, 
					// to rows and columns separately, then construct a bit flag.
					return BitOr(
						((rowShape.size() == 1) ? 
							ApplyShapes::after : ApplyShapes::rowsBefore),
						((colShape.size() == 1) ? 
							ApplyShapes::after : ApplyShapes::colsBefore ));
				}
			}();
			
			// When we apply the shape, we will need a controlling tile.
			// The sources will be local arrays rowShape_l and colShape_l, 
			// so we zero-out row/col_beg.
			TileSpecs const shapeTile(0, tile->num_rows, 0, tile->num_cols);
						
			///////////////////////////////////////////////////////////////
			// Prepare for the l-loop by calculating inter-particle weights and dot-products
			///////////////////////////////////////////////////////////////
			
			// Zero out the working arrays. Only do once per tile because 
			// each tile will always fill these arrays the same way in the l-loop,
			// so any necessary zero-buffer will remain zero.
			Pl_computer.z.fill(real_t(0));
			fProd.fill(real_t(0));
			Hl_accumulate.fill(real_t(0));
			rowShape_l.fill(real_t(0));
			colShape_l.fill(real_t(0));
			
			// Store information about the fill (length of the altered region and
			// fill pattern). All other tiles will be tilled like fProd (since they all share 
			// the same TileSpecs), so we only need to store the TileFill once.
			TileFill const fill = symmetric ? 
				FillTile_Symmetric<Equals>(fProd, *tile, rows->f) : 
				FillTile<Equals>(fProd, *tile, rows->f, cols->f);
				
			// For symmetric tiles, the symmetry factor should only be applied once,
			// and has already been applied to fProd. Thus, p^_i . p^_j proceeds via 
			// the normal FillTile, regardless of symmetry.
			{
				auto const fill_pDot = 
				    FillTile<Equals>(Pl_computer.z, *tile, rows->x, cols->x);
				FillTile<PlusEquals>(Pl_computer.z, *tile, rows->y, cols->y);
				FillTile<PlusEquals>(Pl_computer.z, *tile, rows->z, cols->z);
				
				assert(fill_pDot == fill); // Verify the filling is the same
			}
				
			/* (fill.size) tells us the size of fProd and Pl_computer.z which are altered, 
			 * and this variable will bound the inner product when we fill Hl_accumulate.
			 * Similarly, we only want to sum up the filled portion of Hl_accumulate.
			 * However, because BinaryAccumulate must start with a power-of-2 size, 
			 * we start with the smallest power-of-2 which covers the filled portion.
			*/ 
			size_t const sumSize = kdp::IsPowerOfTwo(fill.size) ? 
				fill.size : 
				2 * kdp::LargestBit(fill.size);
			assert(sumSize <= TileSpecs::incrementSize);
			assert(sumSize >= fill.size);
			
			// Prepare to iterate (telling Pl_computer that we only need to
			// update the filled region of Pl_computer.z).
			Pl_computer.Reset(fill.size);
			
			for(size_t l = 1; l <= lMax; ++l)
			{
				// The branch is quite predictable, and saves one unused call to Pl_computer.Next()
				// (i.e. if we put the call to Pl_computer.Next() at the end of the loop).
				// The speed advantage is actually noticeable.
				if(l > 1)
					Pl_computer.Next();
				assert(Pl_computer.l() == l);
				
				// Do the inner-product for delta-distribution particles
				for(size_t k = 0; k < fill.size; ++k)
					Hl_accumulate[k] = Pl_computer.P_l()[k] * fProd[k];
				
				////////////////////////////////////////////////////////////
				// Multiply Hl_accumulate by particle shape
				////////////////////////////////////////////////////////////
				
				// First cache the numerical values of row and column shapes				
				for(size_t i = 0; i < rowShape.size(); ++i)
					rowShape_l[i] = rowShape[i]->hl(l);
				
				// colShape may be empty if the tile is diagonal, but this loop is safe
				for(size_t j = 0; j < colShape.size(); ++j)
					colShape_l[j] = colShape[j]->hl(l);
				
				// Shapes multiply the existing Hl_accumulate
				switch(applyShapes)
				{
					case ApplyShapes::bothBefore:
												
						if(diagonal)
						{
							// The symmetry factor was already applied to fProd,
							// so we send a symmetry factor of 1
							auto const fill_shape = 
								FillTile_Symmetric<TimesEquals>(Hl_accumulate, 
									shapeTile, rowShape_l, real_t(1));
							assert(fill_shape == fill);
						}
						else
						{
							assert(rowShape.size() == tile->num_rows);
							assert(colShape.size() == tile->num_cols);							
							
							FillTile<TimesEquals>(Hl_accumulate, shapeTile, rowShape_l, colShape_l);
						}
					break;
					
					case ApplyShapes::rowsBefore:
						assert(rowShape.size() == tile->num_rows);
						FillTile_Rows<TimesEquals>(Hl_accumulate, shapeTile, rowShape_l, 
							fill.isTriangular());
					break;
					
					case ApplyShapes::colsBefore:
						assert(colShape.size() == tile->num_cols);
						FillTile_Cols<TimesEquals>(Hl_accumulate, shapeTile, colShape_l, 
							fill.isTriangular());
					break;
					
					default: // Apply shapes after; nothing to do here
					break;
				}
							
				// Having applied all shapes that must come before accumulation, we accumulate
				real_t Hl_sum = kdp::BinaryAccumulate_Destructive(Hl_accumulate, sumSize);				
												
				switch(applyShapes) // Now apply shapes after accumulation
				{
					case ApplyShapes::bothBefore: // Nothing to do, already done
					break;
					
					case ApplyShapes::rowsBefore:
						assert(colShape.size() == 1);
						Hl_sum *= colShape_l[0];
					break;
					
					case ApplyShapes::colsBefore:
						assert(rowShape.size() == 1);
						Hl_sum *= rowShape_l[0];
					break;
					
					case ApplyShapes::after:
						assert(rowShape.size() == 1);
						assert(colShape.size() <= 1);
						Hl_sum *= (diagonal ? 
							kdp::Squared(rowShape_l[0]) : 
							rowShape_l[0] * colShape_l[0]);
					break;
				}
				
				hl_partial[l - 1] += Hl_sum; // l = 0 is not stored
			}// end l-loop
		}// end NextTile loop
	}// end array allocation	
			
	// No more tiles; this thread is done doing major work.
	// Since it has nothing better to do, use the thread to 
	// add to Hl_total before requesting the next job.
	Finalize_Thread(tileCount, hl_partial);
}

////////////////////////////////////////////////////////////////////////
// Job_AngularResolution
////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job_AngularResolution::TileSpecs::TileSpecs
	(size_t const row_beg_in, size_t const num_rows_in,
	size_t const col_beg_in, size_t const num_cols_in, 
	TileType const type_in):
Job::TileSpecs(row_beg_in, num_rows_in, col_beg_in, num_cols_in), 
type(type_in) {}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::AngleWeight::AngleWeight
	(real_t const angle_in, real_t const weight_in):
angle(angle_in), weight(weight_in) {}

////////////////////////////////////////////////////////////////////////

bool PowerSpectrum::AngleWeight::operator < (AngleWeight const& rhs) const
{
	return (this->angle < rhs.angle);
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::AngleWeight::ReSort
	(std::vector<AngleWeight>& headTail, size_t const n_sorted)
{
	assert(n_sorted <= headTail.size());
	std::sort(headTail.begin() + n_sorted, headTail.end());
	
	if(n_sorted > 0)
		std::inplace_merge(headTail.begin(), headTail.begin() + n_sorted, headTail.end());
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job_AngularResolution::Accumulate::operator()
	(std::vector<AngleWeight>& angleWeight_partial)
{
	if(this->empty())
		this->std::vector<AngleWeight>::operator=(std::move(angleWeight_partial));
	else
	{
		size_t const initSize = this->size();
		
		this->insert(this->end(), angleWeight_partial.cbegin(), angleWeight_partial.cend());
	
		// We ASSUME that angleWeight_partial is already sorted
		std::inplace_merge(this->begin(), this->begin() + initSize, this->end());
	}
};

////////////////////////////////////////////////////////////////////////

PowerSpectrum::Job_AngularResolution::Job_AngularResolution(DetectorObservation const& observation, 
	real_t const fInner_scale, bool const xi_cut):
towers(observation.towers), // verbatim copies, no normalization
tracks(observation.tracks)
{
	if(std::fabs(observation.fTotal() - 1.) > 1e-8)
		throw std::runtime_error("PowerSpectrum::AngularResolution: detector observation is not normalized.");
	
	// Integrating \int_0^thetaR dOmega = sin(thetaR/2)**2 = surfaceFraction
	for(real_t const surfaceFraction : observation.towerAreas)
		twrRadii.emplace_back(real_t(2)*std::asin(std::sqrt(surfaceFraction)));
		
	// If all towers share the same radii, simply copy that universal radii
	// (this rote copying simplifies the logic of the thread loops)
	if(twrRadii.size() < towers.size())
		twrRadii.assign(towers.size(), twrRadii.at(0));
	
	if(xi_cut)
	{
		// Expected radius of Voronoi area: 1 / N = sin**(R/2) ==> R = 2 asin(1/sqrt(N));
		// therefore, the expected inter-particle angle is 2R
		real_t const xi_interParticle_expected = 
			real_t(4)*std::asin(real_t(1)/std::sqrt(real_t(observation.size())));
		
		// Expand this expectation by the safety margin (ensure it doesn't get larger than pi)
		xi_max = std::min(real_t(M_PI), angleMargin * xi_interParticle_expected);
		cosXi_min = std::cos(xi_max);
	}
	else
	{
		/* If we don't want a cut on angle, then we proceed in two stages;
		 * (i) intra-particle angles (cosXi = 1) will be mapped to cosXi = -100.
		 * We want to exclude these angles (which is why we must use cosXi > cosXi_min).
		 * After angles are made extensive, we don't want to exclude any,
		 * so we use a non-sensical maximum.
		*/  
		cosXi_min = real_t(-2);
		xi_max = real_t(INFINITY);
	}
	
	/////////////////////////////////////////////////////////////////////
	// Partition the outer products
	
	// TrackTrack
	tileVec = TileSpecs::Partition_Symmetric_ptr<TileSpecs>(tracks.size(), 
		TileType::TrackTrack);
	
	// TrackTower
	{
		// For track-tower, put towers in the rows; this ensures that 
		// we iterate over tower radius in the outer (row) loop.
		auto newTiles = TileSpecs::Partition_ptr<TileSpecs>(towers.size(), tracks.size(),
			TileType::TowerTrack);
			
		// We have to move the newTiles because they are unique_ptr (move-only).
		// std::back_inserter returns an OutputIterator that can insert 
		// new elements at the end of the container (via push_back, c.f. 
		// most OutputIterators cam simply overwrite existing elements). 
		std::move(newTiles.begin(), newTiles.end(), std::back_inserter(tileVec));
		
		// Alternatively, we could resize tileVec, then pass the last valid iterator
		// NOTE: reserving won't work because the vector size must change
		//~ size_t const size_existing = tileVec.size();
		//~ tileVec.resize(size_existing + newTiles.size());
		//~ std::move(newTiles.begin(), newTiles.end(), tileVec.begin() + size_existing);
	}
	
	// TowerTower
	{
		auto newTiles = TileSpecs::Partition_Symmetric_ptr<TileSpecs>(towers.size(), 
			TileType::TowerTower);
			
		std::move(newTiles.begin(), newTiles.end(), std::back_inserter(tileVec));
	}
	
	/////////////////////////////////////////////////////////////////////
	remainingTiles = tileVec.size();
	weight_target = fInner_scale * (towers.fInner() + tracks.fInner());
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job_AngularResolution::DoTiles()
{
	using namespace TiledOuter;
	
	// The dot product of unit vectors must exist in [-1, 1]
	real_t constexpr badDot = real_t(-100);
	
	std::vector<AngleWeight> angleWeight_sorted;	
	size_t tileCount = 0; // How many tiles do we calculate?
	{
		array_t weight; // f_i * f_j (not l-dependent)
		array_t cosXi; // p^_i . p^_j
		
		real_t totalWeight = real_t(0);
		size_t n_sorted = 0;
		TileSpecs* tile; // We request tiles from NextTile() and store them here
		
		// Cast from Job::TileSpecs to Job_AngularResolution::TileSpecs;
		// NextTile() returns nullptr when the tiles are all gone
		while((tile = static_cast<TileSpecs*>(NextTile())))
		{
			++tileCount;
			
			// Zero out the working arrays (because all elements may not be filled).
			weight.fill(real_t(0)); // zero weight cannot alter angular resolution
			cosXi.fill(badDot); // zero IS a valid dot product, so fill to nonsense
			
												GCC_IGNORE_PUSH(-Wreturn-type) // switch covers all options, return guaranteed
			// We store how long the altered region is (fill.size), 
			// and whether the fill is ragged (fill.ragged).
			// All other arrays will be tilled like weight (since they all share 
			// the same TileSpecs), so we only need to store the TileFill once.
			// For symmetric tiles (TrackTrack and TowerTower), 
			// the symmetry factor should only be applied once (to weight). 
			// Thus, p^_i . p^_j proceeds via the normal FillTile, regardless of symmetry.
			TileFill const fill = [&]()
			{
				switch(tile->type)
				{
					case TileType::TrackTrack:
						
						FillTile<Equals>(		cosXi, *tile, tracks.x, tracks.x);
						FillTile<PlusEquals>(cosXi, *tile, tracks.y, tracks.y);
						FillTile<PlusEquals>(cosXi, *tile, tracks.z, tracks.z);
						
						return FillTile_Symmetric<Equals>(weight, *tile, tracks.f);
					break;
					
					case TileType::TowerTrack:
					{						
						FillTile<Equals>(		cosXi, *tile, towers.x, tracks.x);
						FillTile<PlusEquals>(cosXi, *tile, towers.y, tracks.y);
						FillTile<PlusEquals>(cosXi, *tile, towers.z, tracks.z);
						
						auto const thisFill = FillTile<Equals>(weight, *tile, towers.f, tracks.f);
						
						for(size_t k = 0; k < thisFill.size; ++k)
							weight[k] *= real_t(2); // This whole matrix appears twice
					
						return thisFill;
					}
					break;
					
					case TileType::TowerTower:
						
						FillTile<Equals>(		cosXi, *tile, towers.x, towers.x);
						FillTile<PlusEquals>(cosXi, *tile, towers.y, towers.y);
						FillTile<PlusEquals>(cosXi, *tile, towers.z, towers.z);
						
						return FillTile_Symmetric<Equals>(weight, *tile, towers.f);
					break;
				}
			}();								GCC_IGNORE_POP
			
			if(fill.isTriangular())
			{
				size_t k = 0;
				
				for(size_t i = 0; i < tile->num_rows; ++i)
				{
					for(size_t j = 0; j < i; ++j) // Only off-diagonal elements
					{
						if(cosXi[k] > cosXi_min)
						{
							totalWeight += NewAngle(angleWeight_sorted, 
								cosXi[k], weight[k], 
								tile->row_beg + i, tile->col_beg + j, tile->type);
						}
						++k;
					}
					++k; // Skip the diagonal element
				}
				assert(k == kdp::GaussSum(tile->num_rows));
			}
			else
			{
				// 1. If diagonal, zero out the diagonal (simplifies logic of next loop)				
				if((tile->type not_eq TileType::TowerTrack) // symmetric
					and tile->isDiagonal()) // diagonal
				{
					assert(tile->num_cols == tile->num_rows);
					
					// RowMajor and ColMajor both use this loop
					for(size_t i = 0; i < tile->num_rows; ++i)
						cosXi[i * TileSpecs::tileWidth + i] = badDot;
				}
				
				// 2. Check for small angles				
				if(fill.pattern == TileFill::DataPattern::ColMajor)
				{
					for(size_t j = 0; j < tile->num_cols; ++j)
					{
						for(size_t i = 0; i < TileSpecs::tileWidth; ++i)
						{
							size_t const k = j * TileSpecs::tileWidth + i;
							
							if(cosXi[k] > cosXi_min)
							{
								// The raw angle is a candidate; make extensive and
								// append to angleWeight_sorted IFF the extensive angle is 
								// smaller than xi_min; return weight if accepted (0 if rejected).
								totalWeight += NewAngle(angleWeight_sorted, 
									cosXi[k], weight[k],
									tile->row_beg + i, tile->col_beg + j, tile->type);
							}
						}
					}
				}
				else
				{
					for(size_t i = 0; i < tile->num_rows; ++i)
					{
						for(size_t j = 0; j < tile->num_cols; ++j)
						{
							size_t const k = i * TileSpecs::tileWidth + j;
							
							if(cosXi[k] > cosXi_min)
							{
								// The raw angle is a candidate; make extensive and
								// append to angleWeight_sorted IFF the extensive angle is 
								// smaller than xi_min; return weight if accepted (0 if rejected).
								totalWeight += NewAngle(angleWeight_sorted, 
									cosXi[k], weight[k],
									tile->row_beg + i, tile->col_beg + j, tile->type);
							}
						}
					}
				}
			}
			
			AngleWeight::ReSort(angleWeight_sorted, n_sorted); // Sort the tail and merge
			
			if(totalWeight > real_t(2) * weight_target)
				totalWeight -= StripTailWeight(angleWeight_sorted, totalWeight);
			
			n_sorted = angleWeight_sorted.size();
		}
		
		if(totalWeight > weight_target)
			StripTailWeight(angleWeight_sorted, totalWeight);
	}
	Finalize_Thread(tileCount, angleWeight_sorted);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::Job_AngularResolution::NewAngle
	(std::vector<AngleWeight>& angleWeight_sorted,
	real_t const cosXi, real_t const weight,
	size_t const i_abs, size_t const j_abs, TileType const type) const
{
	assert(cosXi < real_t(1));
	assert((real_t(-1) - cosXi) < 1e-8);
			
	real_t xi = std::acos(std::max(cosXi, real_t(-1)));
	
	switch(type)
	{
		case TileType::TrackTrack:				
			// No smearing between well-measured tracks
		break;
		
		case TileType::TowerTrack:
		{	
			real_t const R = twrRadii[i_abs];
			xi = R * SmearedDistance_TrackTower(xi/R);
		}	
		break;
		
		case TileType::TowerTower:
		{
			real_t const R = std::hypot(twrRadii[i_abs], twrRadii[j_abs]);
			xi = R * SmearedDistance_TowerTower(xi/R);
		}
		break;
	}
	
	if(xi < xi_max)
	{
		angleWeight_sorted.emplace_back(xi, weight);
		
		return weight;
	}
	else return real_t(0);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::Job_AngularResolution::StripTailWeight
	(std::vector<AngleWeight>& angleWeight_sorted, real_t const totalWeight) const
{
	real_t tailWeight = real_t(0);
	
	{
		real_t const overShoot = totalWeight - weight_target;
		auto it_angle = angleWeight_sorted.rbegin(); // start from the back
		
		for(;((it_angle not_eq angleWeight_sorted.rend()) and (tailWeight < overShoot)); ++it_angle)
			tailWeight += it_angle->weight;
		
		// We want totalWeight ~= weight_target, but also totalWeight > weight_target
		// So if tailWeight > overShoot, we trimmed off one too many; decrement once.
		if(tailWeight > overShoot) 
		{
			tailWeight -= (--it_angle)->weight;
			assert(tailWeight >= real_t(0));
		}
		
		angleWeight_sorted.resize(angleWeight_sorted.rend() - it_angle);
	}
	
	return tailWeight;
}

////////////////////////////////////////////////////////////////////////

void PowerSpectrum::Job_AngularResolution::Finalize_Thread(size_t const numTiles, 
	std::vector<AngleWeight>& angleWeight_sorted)
{
	Job::Finalize_Thread(numTiles, angleWeight_final, angleWeight_sorted);
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::Job_AngularResolution::Get_AngularResolution()
{
	WaitTillDone();
	
	assert(std::is_sorted(angleWeight_final.cbegin(), angleWeight_final.cend()));
		
	// We use the geometric mean because it averages over scales
	real_t geoMean = real_t(0);
	real_t weight = real_t(0);
	
	/* Add up the smallest angles until their collective weight 
	 * exceeds some weight target (as a function of <f|f>, the approximate
	 * height of the power spectrum's asymptotic plateau). */
	size_t i = 0;
			
	for(; (i < angleWeight_final.size()) and (weight < weight_target); ++i)
	{
		assert(angleWeight_final[i].angle > real_t(0));
		
		real_t const thisWeight = std::min(angleWeight_final[i].weight, weight_target - weight);
		
		geoMean += thisWeight * std::log(angleWeight_final[i].angle);
		weight += thisWeight;
		
		//~ printf("new: %.3e %.3e\n", angleWeight_final[i].angle, thisWeight);
		
		if(thisWeight < angleWeight_final[i].weight)
		{
			assert(kdp::AbsRelError(weight_target, weight) < 1e-8);
			break;
		}
	}
	
	// The loop aborted before we reached our weight target;
	// try again (provided there was a cut on angle)
	if((i == angleWeight_final.size()) and (xi_max < real_t(INFINITY)))
		return real_t(-1);
	else
		return std::exp(geoMean / weight);
}

////////////////////////////////////////////////////////////////////////
//  PowerSpectrum
////////////////////////////////////////////////////////////////////////
		
void PowerSpectrum::Thread()
{
	std::shared_ptr<Job> job;
	
	// Request jobs; Dispatch() will block until there are jobs, 
	// and return nullptr when it's time to die (exit the loop and return)
	while((job = Dispatch()))
		job->DoTiles();
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
		threadPool.emplace_back(&PowerSpectrum::Thread, this);
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

////////////////////////////////////////////////////////////////////////

PowerSpectrum::real_t PowerSpectrum::AngularResolution_Slow(DetectorObservation const& observation,
	real_t const ff_fraction)
{
	observation.CheckValidity_TowerArea();
	
	if(std::fabs(1. - observation.fTotal()) > 1e-8)
		throw std::runtime_error("PowerJects::AngularResolution_Slow: ensemble not normalized");
	
	if(observation.size() < 2)
	{
		return INFINITY;
	}
	else
	{
		std::vector<real_t> twrRadii;
		
		// Integrating \int_0^thetaR dOmega = 
		// 2 pi (1-cos(thetaR)) = Omega ==> sin(thetaR/2)**2 = surfaceFraction
		for(real_t const surfaceFraction : observation.towerAreas)
			twrRadii.emplace_back(real_t(2)*std::asin(std::sqrt(surfaceFraction)));
		
		// If all towers share the same radii, simply copy that universal radii
		// (this rote copying simplifies the logic of the inner loop)
		if(twrRadii.size() < observation.towers.size())
			twrRadii.assign(observation.towers.size(), twrRadii.at(0));
			
		// Create a list of inter-particle angles and correlation weight = f_i * f_j
		std::vector<AngleWeight> angleVec;
		angleVec.reserve(kdp::GaussSum(observation.size() - 1));
				
		// Find the exact angular resolution (memory intensive)		
		for(auto trk = observation.tracks.cbegin(); trk not_eq observation.tracks.cend(); ++trk)
		{
			// We only calculate the lower-half of the symmetric inter-particle matrix
			for(auto trk_other = observation.tracks.cbegin(); trk_other not_eq trk; ++trk_other)
			{
				angleVec.emplace_back(
					// No smearing between tracks; assume very well measured
					trk->pHat.InteriorAngle(trk_other->pHat),
						trk->f * trk_other->f);
			}
			
			for(size_t twr_i = 0; twr_i < observation.towers.size(); ++twr_i)
			{
				// We calculate the dimensionless smeared angle, then rescale it by the tower radius
				angleVec.emplace_back(
					twrRadii[twr_i] * SmearedDistance_TrackTower(
						trk->pHat.InteriorAngle(observation.towers[twr_i].pHat)/twrRadii[twr_i]),
					trk->f * observation.towers[twr_i].f);
			}
		}
			
		for(size_t twr_i = 0; twr_i < observation.towers.size(); ++twr_i)
		{
			// We only calculate the lower-half of the symmetric inter-particle matrix
			for(size_t twr_j = 0; twr_j < twr_i; ++twr_j)
			{
				// For two towers, we use their shared radius
				real_t const twoTowerRadius = std::hypot(twrRadii[twr_i], twrRadii[twr_j]);
				
				angleVec.emplace_back(
					twoTowerRadius	* SmearedDistance_TowerTower(
						observation.towers[twr_i].pHat.InteriorAngle(observation.towers[twr_j].pHat)/twoTowerRadius),
					observation.towers[twr_i].f * observation.towers[twr_j].f);
			}
		}
		
		std::sort(angleVec.begin(), angleVec.end());
		
		/* Now we add up the smallest angles until their collective weight 
		 * exceeds some weight target (as a function of <f|f>, the approximate
		 * height of the power spectrum's asymptotic plateau).
		 * We scale weight_target by 1/2 because each angle/weight appears 
		 * twice in the full matrix, but we only calculated the lower half
		*/ 
		real_t const weight_target = 0.5 * ff_fraction * observation.fInner(); 
		
		// We use the geometric mean because it averages over scales
		real_t geoMean = real_t(0);
		real_t weight = real_t(0);
		
		for(size_t i = 0; (i < angleVec.size()) and (weight < weight_target); ++i)
		{
			real_t const thisWeight = std::min(angleVec[i].weight, weight_target - weight);
			
			geoMean += thisWeight * std::log(angleVec[i].angle);
			weight += thisWeight;
			
			//~ printf("old: %.3e %.3e\n", angleVec[i].angle, 2.*thisWeight);
			
			if(thisWeight < angleVec[i].weight)
			{
				assert(kdp::AbsRelError(weight_target, weight) < 1e-8);
				break;
			}
		}
		
		return std::exp(geoMean / weight);
	}
}

////////////////////////////////////////////////////////////////////////'
		
std::vector<PowerSpectrum::real_t> PowerSpectrum::Launch_Hl_Job(size_t const lMax,
	ShapedParticleContainer const& left, ShapedParticleContainer const& right)
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
	if((lMax == 0) or (left.size() == 0) or (right.size() == 0))
		return std::vector<real_t>();
	else
	{
		std::shared_ptr<Job_Hl> job = std::make_shared<Job_Hl>(lMax, left, right);
		
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

PowerSpectrum::real_t PowerSpectrum::AngularResolution(DetectorObservation const& observation, 
	real_t const fInner_scale)
{
	// Synchronize the thread pool to check that keepAlive is true
	std::unique_lock<std::mutex> threadSafe(threadSafe_lock);
		
	// By using a unique_lock, threadSafe will be unlocked when this exception is thrown
	if(not keepAlive)
		throw std::runtime_error("PowerSpectrum: this object is in the process of being deconstructed!");
	
	// The thread pool is now guaranteed to persist till after the job is complete
	++activeJobs;
	threadSafe.unlock();
		
	// Return a nonsense value if there are no particles
	if(observation.size() == 0)
		return INFINITY;
	else
	{
		// true = cut on angle
		auto job = std::make_shared<Job_AngularResolution>(observation, fInner_scale, true);
		
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
			
		real_t resolution = job->Get_AngularResolution();
		
		// When we didn't keep enough angles to reach weight_target, 
		// try again without throwing any angles away. I expect this will 
		// occur only when the event is small.
		if(resolution < real_t(0))
		{
			// false = consider all angles without a cut
			job = std::make_shared<Job_AngularResolution>(observation, fInner_scale, false);
			
			{
				std::lock_guard<std::mutex> dispatch(dispatch_lock);
					jobQueue.push_back(job);
			}
			
			if(job->RemainingTiles() <= minTilesPerThread)
				newJob.notify_one();
			else
				newJob.notify_all();
				
			resolution = job->Get_AngularResolution();
		}
		assert(resolution >= real_t(0));
		
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
		
		return resolution;
	}
}	
	
////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Obs(size_t const lMax, 
	ShapedParticleContainer const& particles)
{
	return Launch_Hl_Job(lMax, particles, particles);
}

////////////////////////////////////////////////////////////////////////

std::vector<PowerSpectrum::real_t> PowerSpectrum::Hl_Jet(size_t const lMax, 
	ShapedParticleContainer const& jets, std::vector<real_t> const& hl_detector_Filter)
{
	std::vector<real_t> Hl_vec = Launch_Hl_Job(lMax, jets, jets);
	
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
		Hl_jets_particles_future = std::async(std::launch::async, &PowerSpectrum::Launch_Hl_Job, this,
			lMax, std::cref(jets), std::cref(particles));
			
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
	threadReturn.push_back(std::async(std::launch::async, &PowerSpectrum::Thread, this));
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
